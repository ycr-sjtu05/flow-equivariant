import argparse
import math
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# =============================================================================
# 1. DATASET: Moving MNIST
# =============================================================================

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root='./data',
        train=True,
        seq_len=20,
        image_size=64,
        velocity_range_x=(-2, 2),
        velocity_range_y=(-2, 2),
        num_digits=2,
        download=True,
        random_gen=True,
        seed=42
    ):
        super().__init__()
        self.mnist = MNIST(root=root, train=train, download=download)
        self.seq_len = seq_len
        self.image_size = image_size
        self.vx_range = velocity_range_x
        self.vy_range = velocity_range_y
        self.num_digits = num_digits
        self.to_tensor = ToTensor()
        self.random_gen = random_gen
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        imgs = []
        for _ in range(self.num_digits):
            idx = np.random.randint(0, len(self.mnist)) if self.random_gen else self.rng.randint(0, len(self.mnist))
            img_pil, _ = self.mnist[idx]
            img = self.to_tensor(img_pil) 
            imgs.append(img)

        velocities_x, velocities_y = [], []
        positions_x, positions_y = [], []

        for _ in range(self.num_digits):
            if self.random_gen:
                vx = np.random.randint(self.vx_range[0], self.vx_range[1] + 1)
                vy = np.random.randint(self.vy_range[0], self.vy_range[1] + 1)
                x0 = np.random.randint(0, self.image_size)
                y0 = np.random.randint(0, self.image_size)
            else:
                vx = self.rng.randint(self.vx_range[0], self.vx_range[1] + 1)
                vy = self.rng.randint(self.vy_range[0], self.vy_range[1] + 1)
                x0 = self.rng.randint(0, self.image_size)
                y0 = self.rng.randint(0, self.image_size)
            
            velocities_x.append(vx)
            velocities_y.append(vy)
            positions_x.append(x0)
            positions_y.append(y0)

        seq = torch.zeros(self.seq_len, 1, self.image_size, self.image_size)

        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)
            for i, img in enumerate(imgs):
                pad_h = (self.image_size - 28) // 2
                pad_w = (self.image_size - 28) // 2
                padded = torch.zeros(1, self.image_size, self.image_size)
                padded[:, pad_h:pad_h+28, pad_w:pad_w+28] = img

                shift_x = positions_x[i] + velocities_x[i] * t
                shift_y = positions_y[i] + velocities_y[i] * t
                
                moved = torch.roll(padded, shifts=(shift_y, shift_x), dims=(1, 2))
                frame += moved
            
            seq[t] = frame.clamp(0.0, 1.0)

        return seq, 0 

# =============================================================================
# 2. MODEL: Flow-Equivariant Mamba (Stabilized)
# =============================================================================

class FlowMambaCell(nn.Module):
    def __init__(self, d_model, d_state, v_range, image_size, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.image_size = image_size
        
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        # Precomputed Gather Indices
        gather_indices = []
        H, W = image_size, image_size
        for (vx, vy) in self.v_list:
            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            src_y = (y_grid - vy) % H
            src_x = (x_grid - vx) % W
            flat_idx = src_y * W + src_x
            gather_indices.append(flat_idx)
        gather_indices = torch.stack(gather_indices, dim=0)
        self.register_buffer('gather_indices', gather_indices.view(1, self.num_v, 1, 1, H*W))

        # --- Stability: GroupNorm for Input ---
        # Normalizing the input u_t before it generates parameters prevents signal explosion
        self.norm = nn.GroupNorm(4, d_model)

        pad = kernel_size // 2
        self.conv_delta = nn.Conv2d(d_model, d_model, kernel_size, padding=pad, padding_mode='circular', bias=True)
        self.conv_B = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)
        self.conv_C = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)

        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_A_real = nn.Parameter(log_A_real)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1)))

    def forward(self, u_t, s_prev):
        B, D, H, W = u_t.shape
        N = self.d_state
        V = self.num_v
        
        # Stability: Normalize input to the cell
        u_t_norm = self.norm(u_t)

        # 1. Parameter Generation
        # Stability: Clamp delta prevents exp() overflow
        delta = F.softplus(self.conv_delta(u_t_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=1e-4, max=5.0) 

        B_val = self.conv_B(u_t_norm)
        C_val = self.conv_C(u_t_norm)

        # 2. Discretization
        # Force A to be negative -> Stability: System cannot explode (eigenvalues < 1)
        A = -torch.exp(self.log_A_real) 
        
        delta_bc = delta.unsqueeze(2)  
        A_bc = A.view(1, D, N, 1, 1)   
        B_val_bc = B_val.unsqueeze(1)  
        
        A_bar = torch.exp(delta_bc * A_bc) 
        B_bar = delta_bc * B_val_bc 

        # 3. Vectorized Transport
        s_flat = s_prev.view(B, V, D, N, H*W)
        indices = self.gather_indices.expand(B, V, D, N, H*W)
        s_transported_flat = torch.gather(s_flat, -1, indices)
        s_transported = s_transported_flat.view(B, V, D, N, H, W)

        # 4. SSM Update
        u_t_bc = u_t.view(B, 1, D, 1, H, W)
        s_new = A_bar.unsqueeze(1) * s_transported + B_bar.unsqueeze(1) * u_t_bc

        # 5. Readout
        C_bc = C_val.view(B, 1, 1, N, H, W)
        y_state = torch.sum(s_new * C_bc, dim=3)
        y_skip = u_t.unsqueeze(1) * self.D.view(1, 1, D, 1, 1)
        y_new = y_state + y_skip
        
        return y_new, s_new


class FlowMambaModel(nn.Module):
    def __init__(self, input_channels=1, d_model=32, d_state=8, image_size=64, v_range=1, encoder_layers=2, decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.v_range = v_range
        self.num_v = (2*v_range + 1)**2
        
        # Encoder with Normalization
        enc = []
        for i in range(encoder_layers):
            in_c = input_channels if i == 0 else d_model
            enc.append(nn.Conv2d(in_c, d_model, 3, padding=1, padding_mode='circular'))
            # Stability: GroupNorm helps train deep convs without explosion
            enc.append(nn.GroupNorm(4, d_model)) 
            enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        
        self.cell = FlowMambaCell(d_model, d_state, v_range, image_size)
        
        # Decoder
        dec = []
        for i in range(decoder_layers):
            dec.append(nn.Conv2d(d_model, d_model, 3, padding=1, padding_mode='circular'))
            dec.append(nn.GroupNorm(4, d_model)) # Stability
            dec.append(nn.ReLU())
        dec.append(nn.Conv2d(d_model, input_channels, 3, padding=1, padding_mode='circular'))
        self.decoder = nn.Sequential(*dec)

    def forward(self, input_seq, pred_len=10, teacher_forcing_ratio=0.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        s = torch.zeros(batch, self.num_v, self.d_model, self.d_state, H, W, device=device)
        
        for t in range(T_in):
            x_t = input_seq[:, t]
            u_t = self.encoder(x_t)
            _, s = self.cell(u_t, s)
            
        outputs = []
        last_frame = input_seq[:, -1]
        
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                inp_t = target_seq[:, t]
            else:
                inp_t = last_frame
                
            u_t = self.encoder(inp_t)
            y_lifted, s = self.cell(u_t, s)
            
            y_pooled, _ = torch.max(y_lifted, dim=1)
            
            pred_frame = self.decoder(y_pooled)
            
            # Stability: Bound the output to [0, 1] using Sigmoid
            # This prevents MSE Loss from generating massive gradients if predictions drift.
            pred_frame = torch.sigmoid(pred_frame)
            
            outputs.append(pred_frame)
            last_frame = pred_frame
            
        return torch.stack(outputs, dim=1)

# =============================================================================
# 3. MAIN EXECUTION
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
        
        # Double check for NaN before backward (Optional debug)
        if torch.isnan(loss):
            print("NaN Loss detected! Skipping batch.")
            continue

        loss.backward()
        
        # Stability: Clip gradients globally
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
    # TIP: With 6-8 GPUs, you should increase this. Try 64, 128, or 256.
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--v_range', type=int, default=1, help='Range of velocities')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=8)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=20)
    args = parser.parse_args()

    # 1. Detect Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} with {num_gpus} GPUs available.")
    
    # 2. Setup Dataset
    dataset = MovingMNISTDataset(
        root='./data', seq_len=args.seq_len, image_size=args.image_size,
        velocity_range_x=(-args.v_range, args.v_range),
        velocity_range_y=(-args.v_range, args.v_range)
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # IMPORTANT: With DataParallel, this batch_size is divided among GPUs.
    # If batch_size=16 and you have 8 GPUs, each GPU gets 2 samples.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Initialize Model
    model = FlowMambaModel(
        input_channels=1,
        d_model=args.d_model, 
        d_state=args.d_state,
        image_size=args.image_size,
        v_range=args.v_range
    ).to(device)

    # --- KEY CHANGE: Wrap model for Multi-GPU ---
    if num_gpus > 1:
        print(f"Parallelizing model across {num_gpus} GPUs...")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Start Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        # Train & Eval
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.input_frames)
        val_loss = evaluate(model, val_loader, criterion, device, args.input_frames)
        
        end = time.time()
        
        print(f"Epoch {epoch} | Time: {end-start:.1f}s | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # --- KEY CHANGE: Save logic ---
        if epoch % 5 == 0:
            # If wrapped in DataParallel, we want to save 'model.module'
            # so the weights can be loaded easily without DataParallel later.
            if isinstance(model, nn.DataParallel):
                state_to_save = model.module.state_dict()
            else:
                state_to_save = model.state_dict()
            
            torch.save(state_to_save, f"flow_mamba_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()