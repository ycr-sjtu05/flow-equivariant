import argparse
import math
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# =============================================================================
# 1. DATASET: Rotating MNIST
# =============================================================================

class RotatingMNISTDataset(Dataset):
    """
    Generates sequences of rotating MNIST digits on the fly.
    """
    def __init__(
        self,
        root='./data',
        train=True,
        seq_len=20,
        image_size=64, # Increased to 64 to avoid corner cutting during rotation
        angular_velocities=[-40, -20, 0, 20, 40], # Degrees per frame
        num_digits=2,
        download=True,
        random_gen=True,
        seed=42
    ):
        super().__init__()
        self.mnist = MNIST(root=root, train=train, download=download)
        self.seq_len = seq_len
        self.image_size = image_size
        self.angular_velocities = angular_velocities
        self.num_digits = num_digits
        self.to_tensor = ToTensor()
        self.random_gen = random_gen
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        imgs = []
        for _ in range(self.num_digits):
            if self.random_gen:
                idx = np.random.randint(0, len(self.mnist))
            else:
                idx = self.rng.randint(0, len(self.mnist))
            img_pil, _ = self.mnist[idx]
            img = self.to_tensor(img_pil) 
            imgs.append(img)

        # Sample angular velocities and initial angles
        omegas = []
        thetas = []
        
        for _ in range(self.num_digits):
            if self.random_gen:
                w = np.random.choice(self.angular_velocities)
                th = np.random.uniform(0, 360)
            else:
                w = self.rng.choice(self.angular_velocities)
                th = self.rng.uniform(0, 360)
            omegas.append(w)
            thetas.append(th)

        seq = torch.zeros(self.seq_len, 1, self.image_size, self.image_size)

        # Pre-pad digit to avoid cutting corners before placing in canvas
        # MNIST is 28x28. Diagonal is ~40. 
        
        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)
            
            for i, img in enumerate(imgs):
                # 1. Create Affine Matrix for Rotation
                # Angle in radians. Negative because grid_sample rotates grid, not image
                angle = -(thetas[i] + omegas[i] * t) 
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                
                # 2. Pad image to canvas size first (center digit)
                # We place digit in center of a canvas, then rotate the canvas
                pad_h = (self.image_size - 28) // 2
                pad_w = (self.image_size - 28) // 2
                centered = torch.zeros(1, self.image_size, self.image_size)
                centered[:, pad_h:pad_h+28, pad_w:pad_w+28] = img

                # 3. Rotate using grid_sample
                # Matrix for rotation around center
                rot_mat = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0]
                ], dtype=torch.float32).unsqueeze(0) # (1, 2, 3)
                
                grid = F.affine_grid(rot_mat, centered.unsqueeze(0).size(), align_corners=False)
                rotated = F.grid_sample(centered.unsqueeze(0), grid, align_corners=False, padding_mode='zeros').squeeze(0)
                
                frame += rotated

            seq[t] = frame.clamp(0.0, 1.0)

        return seq, 0 

# =============================================================================
# 2. MODEL: Rotation Flow-Equivariant Mamba
# =============================================================================

class RotationTransport(nn.Module):
    """
    Handles the "Flow" part: Rotates the hidden state bank.
    Uses Vectorized Affine Grid Sampling.
    """
    def __init__(self, v_list, image_size):
        super().__init__()
        self.v_list = v_list # List of angular velocities (degrees)
        self.num_v = len(v_list)
        self.image_size = image_size
        
        # Precompute rotation matrices for one time-step (dt=1)
        matrices = []
        for w in v_list:
            # We want to sample from the "past" position.
            # If velocity is w, previous pixel is at -w rotation.
            angle_rad = math.radians(-w) # Inverse flow transport
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            # PyTorch affine matrix: [[cos, -sin, tx], [sin, cos, ty]]
            mat = torch.tensor([[c, -s, 0], [s, c, 0]], dtype=torch.float32)
            matrices.append(mat)
            
        # Stack -> (num_v, 2, 3)
        self.register_buffer('rot_mats', torch.stack(matrices, dim=0))

    def forward(self, s_prev):
        """
        s_prev: (B, V, D, N_ssm, H, W)
        Returns: s_transported (same shape)
        """
        B, V, D, N, H, W = s_prev.shape
        
        # Merge dims for efficient grid sampling
        # We effectively treat (B, D, N) as "Channels" and V as "Batch" for the grid generation
        # Target: Apply rot_mat[i] to s_prev[:, i]
        
        # Reshape to (B*V, Channels, H, W)
        # Channels = D * N
        s_flat = s_prev.view(B * V, D * N, H, W)
        
        # Prepare matrices: Expand (V, 2, 3) -> (B, V, 2, 3) -> (B*V, 2, 3)
        mats_expanded = self.rot_mats.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * V, 2, 3)
        
        # Generate Grid
        grid = F.affine_grid(mats_expanded, s_flat.size(), align_corners=False)
        
        # Sample (Rotate)
        s_rotated_flat = F.grid_sample(s_flat, grid, align_corners=False, padding_mode='zeros')
        
        # Reshape back
        s_transported = s_rotated_flat.view(B, V, D, N, H, W)
        
        return s_transported

class FlowMambaRotCell(nn.Module):
    def __init__(self, d_model, d_state, v_list, image_size, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.v_list = v_list
        self.num_v = len(v_list)
        
        # Rotation Transport Module
        self.transport = RotationTransport(v_list, image_size)

        # Normalization for Stability
        self.norm = nn.GroupNorm(4, d_model)

        # Parameter Generators
        pad = kernel_size // 2
        self.conv_delta = nn.Conv2d(d_model, d_model, kernel_size, padding=pad, bias=True)
        self.conv_B = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, bias=False)
        self.conv_C = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, bias=False)

        # SSM Parameters
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1)))

    def forward(self, u_t, s_prev):
        """
        u_t: (B, D, H, W)
        s_prev: (B, V, D, N, H, W)
        """
        B, D, H, W = u_t.shape
        N = self.d_state
        
        # 1. Parameter Generation
        u_norm = self.norm(u_t)
        
        delta = F.softplus(self.conv_delta(u_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=1e-4, max=5.0) # Stability clamp
        
        B_val = self.conv_B(u_norm)
        C_val = self.conv_C(u_norm)

        # 2. Discretization (Euler)
        A = -torch.exp(self.log_A_real) # Force Negative
        
        # Broadcast prep
        delta_bc = delta.unsqueeze(2) # (B, D, 1, H, W)
        A_bc = A.view(1, D, N, 1, 1)
        B_val_bc = B_val.unsqueeze(1) # (B, 1, N, H, W)
        
        A_bar = torch.exp(delta_bc * A_bc)
        B_bar = delta_bc * B_val_bc # Euler discretization avoids 1/A singularity

        # 3. Transport (The Core Flow Step)
        s_transported = self.transport(s_prev)

        # 4. SSM Update
        u_t_bc = u_t.view(B, 1, D, 1, H, W)
        
        # s_new = A_bar * s_trans + B_bar * u
        s_new = A_bar.unsqueeze(1) * s_transported + B_bar.unsqueeze(1) * u_t_bc

        # 5. Readout
        C_bc = C_val.view(B, 1, 1, N, H, W)
        y_state = torch.sum(s_new * C_bc, dim=3) # Sum over N
        y_skip = u_t.unsqueeze(1) * self.D.view(1, 1, D, 1, 1)
        
        y_new = y_state + y_skip # (B, V, D, H, W)
        
        return y_new, s_new

class FlowMambaRotModel(nn.Module):
    def __init__(self, input_channels=1, d_model=32, d_state=8, image_size=64, 
                 v_list=[-40, -20, 0, 20, 40], encoder_layers=2, decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_v = len(v_list)
        
        # Encoder
        enc = []
        for i in range(encoder_layers):
            in_c = input_channels if i == 0 else d_model
            enc.append(nn.Conv2d(in_c, d_model, 3, padding=1))
            enc.append(nn.GroupNorm(4, d_model))
            enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        
        # Cell
        self.cell = FlowMambaRotCell(d_model, d_state, v_list, image_size)
        
        # Decoder
        dec = []
        for i in range(decoder_layers):
            dec.append(nn.Conv2d(d_model, d_model, 3, padding=1))
            dec.append(nn.GroupNorm(4, d_model))
            dec.append(nn.ReLU())
        dec.append(nn.Conv2d(d_model, input_channels, 3, padding=1))
        self.decoder = nn.Sequential(*dec)

    def forward(self, input_seq, pred_len=10, teacher_forcing_ratio=0.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # Init State
        s = torch.zeros(batch, self.num_v, self.d_model, self.cell.d_state, H, W, device=device)
        
        # Warmup
        for t in range(T_in):
            u_t = self.encoder(input_seq[:, t])
            _, s = self.cell(u_t, s)
            
        outputs = []
        last_frame = input_seq[:, -1]
        
        # Prediction
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                inp_t = target_seq[:, t]
            else:
                inp_t = last_frame
                
            u_t = self.encoder(inp_t)
            y_lifted, s = self.cell(u_t, s)
            
            # Pooling: Invariant Readout (Max over V)
            y_pooled, _ = torch.max(y_lifted, dim=1)
            
            # Decode
            pred_logits = self.decoder(y_pooled)
            pred_frame = torch.sigmoid(pred_logits) # Bound to [0,1]
            
            outputs.append(pred_frame)
            last_frame = pred_frame
            
        return torch.stack(outputs, dim=1)

# =============================================================================
# 3. TRAINING UTILS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, input_frames):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for seq, _ in pbar:
        seq = seq.to(device)
        inp = seq[:, :input_frames]
        target = seq[:, input_frames:]
        
        optimizer.zero_grad()
        preds = model(inp, pred_len=target.shape[1], target_seq=target, teacher_forcing_ratio=0.0)
        
        loss = criterion(preds, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Training | Loss: {loss.item():.5f}")
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, input_frames):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, _ in loader:
            seq = seq.to(device)
            inp = seq[:, :input_frames]
            target = seq[:, input_frames:]
            
            preds = model(inp, pred_len=target.shape[1])
            loss = criterion(preds, target)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4) 
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=8)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--num_digits', type=int, default=2)
    
    # --- ADDED: Argument Parser for v_list ---
    parser.add_argument('--v_list', nargs='+', type=int, 
                        default=[-40, -20, 0, 20, 40],
                        help='List of angular velocities in degrees')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use the parsed v_list
    v_list = args.v_list
    print(f"Rotation Flow-Mamba | V={v_list}")

    # Data
    dataset = RotatingMNISTDataset(
        root='./data', 
        seq_len=args.seq_len, 
        image_size=args.image_size,
        angular_velocities=v_list,
        num_digits=args.num_digits
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,num_workers=8, pin_memory=True)

    # ================= 找到这部分并替换 =================
    # Model
    model = FlowMambaRotModel(
        input_channels=1,
        d_model=args.d_model,
        d_state=args.d_state,
        image_size=args.image_size,
        v_list=v_list
    ) # 先不要 .to(device)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # device_ids 默认就是所有可见的卡
        model = nn.DataParallel(model)

    model = model.to(device)
    # ===================================================

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Start Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.input_frames)
        val_loss = evaluate(model, val_loader, criterion, device, args.input_frames)
        end = time.time()
        
        print(f"Epoch {epoch} | Time: {end-start:.1f}s | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if epoch % 5 == 0:
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, f"flow_mamba_rot_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()