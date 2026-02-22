import argparse
import math
import os
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# =============================================================================
# 1. DATASET: Moving MNIST (On-the-fly Generation)
# =============================================================================

class MovingMNISTDataset(Dataset):
    """
    Generates sequences of moving MNIST digits with wrap-around boundary conditions.
    """
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
        # We can define an arbitrary epoch length since we generate on the fly.
        # Using MNIST size is a reasonable convention.
        return len(self.mnist)

    def __getitem__(self, index):
        # 1. Sample Digits
        imgs = []
        for _ in range(self.num_digits):
            idx = np.random.randint(0, len(self.mnist)) if self.random_gen else self.rng.randint(0, len(self.mnist))
            img_pil, _ = self.mnist[idx]
            img = self.to_tensor(img_pil)  # (1, 28, 28)
            imgs.append(img)

        # 2. Sample Velocities and Positions
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

        # 3. Generate Sequence via Circular Rolling
        seq = torch.zeros(self.seq_len, 1, self.image_size, self.image_size)

        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)
            for i, img in enumerate(imgs):
                # Pad/Resize digit to canvas size
                # MNIST is 28x28. If canvas is larger, we center pad.
                pad_h = (self.image_size - 28) // 2
                pad_w = (self.image_size - 28) // 2
                # Simple placement logic: pad to image_size
                padded = torch.zeros(1, self.image_size, self.image_size)
                padded[:, pad_h:pad_h+28, pad_w:pad_w+28] = img

                # Calculate shift
                shift_x = positions_x[i] + velocities_x[i] * t
                shift_y = positions_y[i] + velocities_y[i] * t
                
                # Apply Flow (Circular Shift)
                moved = torch.roll(padded, shifts=(shift_y, shift_x), dims=(1, 2))
                frame += moved
            
            # Clip overlap
            seq[t] = frame.clamp(0.0, 1.0)

        return seq, 0  # 0 is dummy label


# =============================================================================
# 2. MODEL: Flow-Equivariant Mamba
# =============================================================================

class FlowMambaCell(nn.Module):
    """
    The core recurrent unit implementing the Flow-Equivariant Scan.
    
    Structure:
    - Input Lifting: Trivial (broadcast)
    - Parameter Generation: Conv2d(u_t) -> Delta, B, C
    - Discretization: ZOH (Pointwise)
    - Transport: torch.roll (Inverse Flow)
    - Update: s_t = A_bar * transport(s_prev) + B_bar * u_t
    """
    def __init__(self, d_model, d_state, v_range, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Velocity Grid V
        # v_range=2 -> {-2, -1, 0, 1, 2} x {-2, ...} -> 25 velocities
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        # 1. Parameter Generators (Selective Fields)
        # We use 2D circular convolutions to generate parameters from the spatial input u_t
        pad = kernel_size // 2
        
        # Project u_t -> Delta (Time scale)
        self.conv_delta = nn.Conv2d(d_model, d_model, kernel_size, padding=pad, padding_mode='circular', bias=True)
        
        # Project u_t -> B (Input Dynamics)
        self.conv_B = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)
        
        # Project u_t -> C (Output Dynamics)
        self.conv_C = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)

        # 2. Learnable SSM Parameters
        # A: (d_model, d_state) - Diagonal State Matrix (S4D initialization)
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_A_real = nn.Parameter(log_A_real)
        
        # D: (d_model) - Skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Delta bias (softplus initialization shift)
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1))) # init to dt=1 roughly

    def forward(self, u_t, s_prev):
        """
        Args:
            u_t: (B, d_model, H, W) - Spatial Input at time t
            s_prev: (B, num_v, d_model, d_state, H, W) - Hidden State at time t-1
            
        Returns:
            y_t: (B, num_v, d_model, H, W) - Output at time t
            s_t: (B, num_v, d_model, d_state, H, W) - Updated Hidden State
        """
        batch, _, H, W = u_t.shape
        
        # --- A. Generate Parameters (Shared across V) ---
        
        # 1. Delta (Timescale)
        # (B, d_model, H, W)
        delta = F.softplus(self.conv_delta(u_t) + self.dt_inv_softplus)
        
        # 2. B matrix
        # (B, d_state, H, W)
        B_val = self.conv_B(u_t)
        
        # 3. C matrix
        # (B, d_state, H, W)
        C_val = self.conv_C(u_t)

        # --- B. Discretization (Pointwise ZOH) ---
        
        # A is (d_model, d_state). We rely on broadcasting over (B, H, W).
        # A_bar = exp(delta * A)
        # Shape: (B, d_model, d_state, H, W)
        A = -torch.exp(self.log_A_real) # Force A to be negative for stability
        
        # Reshape for broadcasting:
        # delta: (B, D, 1, H, W)
        # A:     (1, D, N, 1, 1)
        delta_bc = delta.unsqueeze(2)
        A_bc = A.view(1, self.d_model, self.d_state, 1, 1)
        
        A_bar = torch.exp(delta_bc * A_bc) # (B, D, N, H, W)
        
        # B_bar = (1/A)*(exp(delta*A)-1) * delta * B
        # Simplification often used: B_bar approx delta * B for small steps, 
        # but let's use the explicit ZOH form for correctness if possible, 
        # or the discretized B form: B_bar = (exp(delta*A) - I) * A^{-1} * B
        # Ideally: B_bar = (A_bar - 1) * A^{-1} * B_val
        # broadcasting B_val: (B, 1, N, H, W)
        inv_A = 1.0 / A_bc
        B_val_bc = B_val.unsqueeze(1) # (B, 1, N, H, W)
        B_bar = (A_bar - 1.0) * inv_A * B_val_bc # (B, D, N, H, W)

        # --- C. Flow-Equivariant Scan (Iterate over V) ---
        
        s_next_list = []
        y_list = []
        
        # Pre-reshape inputs for the loop
        # u_t: (B, D, 1, H, W) for broadcasting against s
        u_t_bc = u_t.unsqueeze(2) 
        
        # Pre-reshape C and D for readout
        C_val_bc = C_val.unsqueeze(1) # (B, 1, N, H, W)
        D_bc = self.D.view(1, self.d_model, 1, 1, 1)

        for i, (vx, vy) in enumerate(self.v_list):
            # 1. Transport (Inverse Flow)
            # The paper defines flow action as psi_t(v).x = x(g - v*t).
            # The recurrence uses psi_1(v)^{-1}.
            # If flow is translation by v, inverse is translation by -v.
            # torch.roll(shifts=(dy, dx))
            # We roll by (-vy, -vx).
            
            s_prev_v = s_prev[:, i] # (B, D, N, H, W)
            
            # Note on Roll: 
            # shifts=(y_shift, x_shift). dims=(-2, -1) are (H, W).
            s_transported = torch.roll(s_prev_v, shifts=(-vy, -vx), dims=(-2, -1))
            
            # 2. SSM Update
            # s_t = A_bar * s_trans + B_bar * u_t
            # All elementwise
            s_new = A_bar * s_transported + B_bar * u_t_bc
            
            # 3. Readout
            # y_t = C * s_t + D * u_t
            # Sum over state dimension N: (B, D, N, H, W) * (B, 1, N, H, W) -> sum dim 2 -> (B, D, H, W)
            y_state = torch.sum(s_new * C_val_bc, dim=2) 
            y_skip = u_t * self.D.view(1, self.d_model, 1, 1)
            y_new = y_state + y_skip
            
            s_next_list.append(s_new)
            y_list.append(y_new)
            
        # Stack results
        s_next = torch.stack(s_next_list, dim=1) # (B, |V|, D, N, H, W)
        y_out = torch.stack(y_list, dim=1)       # (B, |V|, D, H, W)
        
        return y_out, s_next


class FlowMambaModel(nn.Module):
    """
    Full Seq2Seq model using FlowMambaCell.
    """
    def __init__(self, 
                 input_channels=1, 
                 d_model=64, 
                 d_state=16, 
                 image_size=64,
                 v_range=2, 
                 encoder_layers=2,
                 decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.v_range = v_range
        self.num_v = (2*v_range + 1)**2
        
        # Encoder (Spatial Lifting: 1 -> d_model)
        enc = []
        for i in range(encoder_layers):
            in_c = input_channels if i == 0 else d_model
            enc.append(nn.Conv2d(in_c, d_model, 3, padding=1, padding_mode='circular'))
            enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        
        # Flow Mamba Cell
        self.cell = FlowMambaCell(d_model, d_state, v_range)
        
        # Decoder (Projection: d_model -> 1)
        dec = []
        for i in range(decoder_layers):
            dec.append(nn.Conv2d(d_model, d_model, 3, padding=1, padding_mode='circular'))
            dec.append(nn.ReLU())
        dec.append(nn.Conv2d(d_model, input_channels, 3, padding=1, padding_mode='circular'))
        self.decoder = nn.Sequential(*dec)

    def forward(self, input_seq, pred_len=10, teacher_forcing_ratio=0.0, target_seq=None):
        """
        input_seq: (B, T_in, C, H, W)
        """
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # Initialize Hidden State
        # s: (B, |V|, d_model, d_state, H, W)
        s = torch.zeros(batch, self.num_v, self.d_model, self.d_state, H, W, device=device)
        
        # 1. Process Input Sequence (Encoder)
        for t in range(T_in):
            x_t = input_seq[:, t] # (B, C, H, W)
            u_t = self.encoder(x_t) # (B, D, H, W)
            
            # Recurrence
            y_lifted, s = self.cell(u_t, s)
            # We don't use encoder outputs for prediction, just state warmup
            
        # 2. Prediction Loop (Decoder)
        outputs = []
        last_frame = input_seq[:, -1]
        
        for t in range(pred_len):
            # Input to next step
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                inp_t = target_seq[:, t]
            else:
                inp_t = last_frame
                
            u_t = self.encoder(inp_t)
            
            # Recurrence
            y_lifted, s = self.cell(u_t, s)
            
            # Pooling (Invariant Readout)
            # Max pool over velocity dimension V (dim 1)
            y_pooled, _ = torch.max(y_lifted, dim=1) # (B, D, H, W)
            
            # Decode
            pred_frame = self.decoder(y_pooled)
            outputs.append(pred_frame)
            
            last_frame = pred_frame
            
        return torch.stack(outputs, dim=1) # (B, T_pred, C, H, W)


# =============================================================================
# 3. TRAINING UTILS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, input_frames):
    model.train()
    total_loss = 0
    
    # TQDM progress bar for real-time loss tracking
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (seq, _) in enumerate(pbar):
        seq = seq.to(device) # (B, T, C, H, W)
        
        inp = seq[:, :input_frames]
        target = seq[:, input_frames:]
        
        optimizer.zero_grad()
        
        # Forward
        preds = model(inp, pred_len=target.shape[1], target_seq=target, teacher_forcing_ratio=0.0)
        
        loss = criterion(preds, target)
        loss.backward()
        
        # Gradient Clipping (Standard for RNN/Mamba)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, input_frames):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, _ in loader:
            seq = seq.to(device)
            inp = seq[:, :input_frames]
            target = seq[:, input_frames:]
            
            preds = model(inp, pred_len=target.shape[1], teacher_forcing_ratio=0.0)
            loss = criterion(preds, target)
            total_loss += loss.item()
            
    return total_loss / len(loader)

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Reduced batch size due to memory')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=32, help='Model hidden dim')
    parser.add_argument('--d_state', type=int, default=8, help='SSM state dim')
    parser.add_argument('--v_range', type=int, default=1, help='Velocity range (e.g. 1 -> 3x3=9 velocities)')
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data
    print("Initializing Dataset...")
    full_dataset = MovingMNISTDataset(
        root=args.data_root, 
        train=True, 
        image_size=args.image_size, 
        seq_len=args.seq_len,
        velocity_range_x=(-args.v_range, args.v_range),
        velocity_range_y=(-args.v_range, args.v_range)
    )
    
    # Simple Split (Use a subset for demo speed if needed)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    print(f"Initializing FlowMamba (V-range={args.v_range}, |V|={(2*args.v_range+1)**2})...")
    model = FlowMambaModel(
        input_channels=1,
        d_model=args.d_model,
        d_state=args.d_state,
        image_size=args.image_size,
        v_range=args.v_range
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 4. Training Loop
    print("Starting Training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.input_frames)
        val_loss = evaluate(model, val_loader, criterion, device, args.input_frames)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} | Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "flow_mamba_best.pth")
            print("  -> Saved Best Model")

if __name__ == "__main__":
    main()