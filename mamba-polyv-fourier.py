import argparse
import math
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# =============================================================================
# 1. DATASET: Moving MNIST (UNCHANGED)
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
# 2. MODEL: Moment-Flow Mamba (ALGORITHM CHANGED)
# =============================================================================

class SpectralDerivative(nn.Module):
    """
    Computes spatial derivatives using FFT for exact equivariance.
    d/dx f = IFFT( i * kx * FFT(f) )
    
    Fixed for DataParallel: Stores frequencies as Real buffers to avoid 
    Complex tensor serialization issues on multi-GPU.
    """
    def __init__(self, h, w):
        super().__init__()
        self.h = h
        self.w = w
        
        # Create frequency grids
        freq_y = torch.fft.fftfreq(h) * 2 * np.pi
        freq_x = torch.fft.fftfreq(w) * 2 * np.pi
        
        # Meshgrid
        ky, kx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Register as FLOAT buffers (Real numbers only)
        # We will multiply by 1j inside forward() to create the complex tensor.
        self.register_buffer('ky', ky)
        self.register_buffer('kx', kx)

    def forward(self, x):
        # x: [Batch, Channels, H, W]
        x_freq = fft.fft2(x)
        
        # Reconstruct the complex derivative kernel on the fly
        # ik_y = i * ky
        # We cast to x_freq.device to ensure safety
        ik_y = 1j * self.ky
        ik_x = 1j * self.kx
        
        # Compute gradients
        grad_y_freq = x_freq * ik_y
        grad_x_freq = x_freq * ik_x
        
        # Inverse FFT
        grad_y = fft.ifft2(grad_y_freq).real
        grad_x = fft.ifft2(grad_x_freq).real
        
        return grad_y, grad_x

class MomentFlowMambaCell(nn.Module):
    def __init__(self, d_model, d_state, image_size, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.image_size = image_size
        
        # --- NEW: Spectral Derivative Module ---
        self.diff = SpectralDerivative(image_size, image_size)
        
        # --- Stability: GroupNorm for Input ---
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
        """
        u_t: [B, D, H, W]
        s_prev: [B, 3, D, N, H, W] 
        """
        B, D, H, W = u_t.shape
        N = self.d_state
        
        u_t_norm = self.norm(u_t)

        # 1. Parameter Generation
        delta = F.softplus(self.conv_delta(u_t_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=1e-4, max=5.0) 

        B_val = self.conv_B(u_t_norm) # [B, N, H, W]
        C_val = self.conv_C(u_t_norm) # [B, N, H, W]

        # 2. Discretization
        A = -torch.exp(self.log_A_real) 
        
        # Fix 1: Ensure dimensions align for [B, D, N, H, W]
        delta_bc = delta.unsqueeze(2)    # [B, D, 1, H, W]
        A_bc = A.view(1, D, N, 1, 1)     # [1, D, N, 1, 1]
        B_val_bc = B_val.unsqueeze(1)    # [B, 1, N, H, W]
        
        A_bar = torch.exp(delta_bc * A_bc) # [B, D, N, H, W]
        B_bar = delta_bc * B_val_bc        # [B, D, N, H, W]

        # 3. MOMENT TRANSPORT
        m0 = s_prev[:, 0] 
        mx = s_prev[:, 1] 
        my = s_prev[:, 2] 
        
        flat_m0 = m0.reshape(B, D*N, H, W)
        grad_y_m0_flat, grad_x_m0_flat = self.diff(flat_m0)
        grad_y_m0 = grad_y_m0_flat.view(B, D, N, H, W)
        grad_x_m0 = grad_x_m0_flat.view(B, D, N, H, W)
        
        m0_transported = m0
        mx_transported = mx - grad_x_m0
        my_transported = my - grad_y_m0
        
        # 4. SSM Update
        # Fix 2: Correct Input Broadcast. u_t is [B, D, H, W], we need [B, D, 1, H, W] to match N dim.
        u_t_bc = u_t.unsqueeze(2) 
        
        m0_new = A_bar * m0_transported + B_bar * u_t_bc
        mx_new = A_bar * mx_transported 
        my_new = A_bar * my_transported 
        
        s_new = torch.stack([m0_new, mx_new, my_new], dim=1)

        # 5. Readout (Moment 0)
        # Fix 3: Correct C broadcast. 
        # C_val is [B, N, H, W]. We want [B, 1, N, H, W] to broadcast over D.
        C_bc = C_val.unsqueeze(1) 
        
        # y_state: Sum over N (dim 2). Result shape [B, D, H, W]
        y_state = torch.sum(m0_new * C_bc, dim=2) 
        
        # y_skip: u_t is [B, D, H, W]. D param is [D].
        y_skip = u_t * self.D.view(1, D, 1, 1)
        
        # Combine and add the "Moment" dimension back for the model to squeeze later
        y_new = (y_state + y_skip).unsqueeze(1) # [B, 1, D, H, W]
        
        return y_new, s_new


class MomentFlowMambaModel(nn.Module):
    def __init__(self, input_channels=1, d_model=32, d_state=8, image_size=64, v_range=1, encoder_layers=2, decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Note: v_range is no longer used for state size! 
        # State size is fixed to 3 moments (0, x, y).
        # We keep the arg to match the interface of the previous experiment.
        
        # Encoder
        enc = []
        for i in range(encoder_layers):
            in_c = input_channels if i == 0 else d_model
            enc.append(nn.Conv2d(in_c, d_model, 3, padding=1, padding_mode='circular'))
            enc.append(nn.GroupNorm(4, d_model)) 
            enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        
        self.cell = MomentFlowMambaCell(d_model, d_state, image_size)
        
        # Decoder
        dec = []
        for i in range(decoder_layers):
            dec.append(nn.Conv2d(d_model, d_model, 3, padding=1, padding_mode='circular'))
            dec.append(nn.GroupNorm(4, d_model)) 
            dec.append(nn.ReLU())
        dec.append(nn.Conv2d(d_model, input_channels, 3, padding=1, padding_mode='circular'))
        self.decoder = nn.Sequential(*dec)

    def forward(self, input_seq, pred_len=10, teacher_forcing_ratio=0.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # INIT STATE: [Batch, 3 Moments, D, N, H, W]
        # Size is constant regardless of v_range!
        s = torch.zeros(batch, 3, self.d_model, self.d_state, H, W, device=device)
        
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
            y_moment_0, s = self.cell(u_t, s)
            
            # Readout directly from 0th moment (No pooling needed)
            # Flatten D dim for decoder if needed, but here y is [B, 1, D, H, W]
            # Squeeze time dim
            y_out = y_moment_0.squeeze(1) 
            
            pred_frame = self.decoder(y_out)
            pred_frame = torch.sigmoid(pred_frame)
            
            outputs.append(pred_frame)
            last_frame = pred_frame
            
        return torch.stack(outputs, dim=1)

# =============================================================================
# 3. MAIN EXECUTION (UNCHANGED)
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
        
        if torch.isnan(loss):
            print("NaN Loss detected! Skipping batch.")
            continue

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
    # Increased default batch size for 8 GPUs (32 per GPU * 8 = 256)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--v_range', type=int, default=1, help='Range of velocities')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=8)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=20)
    args = parser.parse_args()

    # 1. Setup Device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs available: {num_gpus}")
        device = torch.device('cuda:0') # Master device
    else:
        print("No GPU detected. Using CPU.")
        device = torch.device('cpu')
    
    dataset = MovingMNISTDataset(
        root='./data', seq_len=args.seq_len, image_size=args.image_size,
        velocity_range_x=(-args.v_range, args.v_range),
        velocity_range_y=(-args.v_range, args.v_range)
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # 2. Optimized DataLoaders for Multi-GPU
    # num_workers=8 helps load data in parallel to keep GPUs busy
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=8, pin_memory=True
    )

    # Note: Using MomentFlowMambaModel (Spectral/FFT version)
    model = MomentFlowMambaModel(
        input_channels=1,
        d_model=args.d_model, 
        d_state=args.d_state,
        image_size=args.image_size,
        v_range=args.v_range
    )

    # 3. Multi-GPU Wrapper
    # This automatically splits the batch into chunks and sends them to all GPUs
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model with DataParallel on {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    
    model = model.to(device)

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
            # 4. Safe Saving
            # If wrapped in DataParallel, access .module to save clean state_dict
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), f"moment_flow_fft_epoch_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"moment_flow_fft_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()