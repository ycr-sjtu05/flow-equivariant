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
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# =============================================================================
# 1. DATASET: Moving MNIST (Unchanged)
# =============================================================================

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root='./data',
        train=True,
        seq_len=20,
        image_size=64,
        velocity_range_x=(-3, 3), 
        velocity_range_y=(-3, 3),
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
# 2. ENCODER / DECODER (Unchanged)
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.act(out)

class DeepEncoder(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(32),
            nn.Conv2d(32, d_model, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(d_model),
            ResBlock(d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class DeepDecoder(nn.Module):
    def __init__(self, d_model, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(d_model),
            nn.ConvTranspose2d(d_model, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 3. CORE: FLOW-EQUIVARIANT MAMBA (STRICT PDF IMPLEMENTATION)
# =============================================================================

class FlowEquivariantMambaCell(nn.Module):
    """
    Implementation of the Flow-Equivariant Mamba Cell as defined in the provided paper.
    
    Paper References:
    - [cite_start]Lifted State: s_t(v, g) [cite: 24]
    - [cite_start]Parameter Sharing: Phi_Delta, Phi_B, Phi_C are G-equivariant (Conv2d) [cite: 45]
    - [cite_start]Trivial Lift: u_t(v, g) = Phi_in(x_t)(g) [cite: 67]
    - [cite_start]Scan Update: Eq (10) [cite: 79]
    """
    def __init__(self, d_model, d_state, v_range, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # [cite_start]1. Define V (Set of flow generators) [cite: 13]
        # In this task, G is 2D Translation, so V is a set of velocities.
        # We ensure V covers the data's velocity range (e.g., -3 to 3).
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        print(f"FlowEquivariantMambaCell initialized with |V|={self.num_v} velocities.")

        pad = kernel_size // 2
        
        # [cite_start]2. Selective Parameter Fields (Shared across v) [cite: 44, 45]
        # These are G-equivariant maps (Conv2d on images).
        # Projects input u_t -> Delta, B, C
        self.conv_delta = nn.Conv2d(d_model, d_model, kernel_size, padding=pad, padding_mode='circular', bias=True)
        self.conv_B = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)
        self.conv_C = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)
        
        # D is usually a skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
        # [cite_start]Learnable A matrix (Continuous time) [cite: 56]
        # Structured as diagonal for efficiency
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_A_real = nn.Parameter(log_A_real)
        
        # Delta bias initialization
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1))) 

    def forward(self, u_t, s_prev):
        """
        Args:
            [cite_start]u_t: Input field x_t (Batch, d_model, H, W) -> Acts as the Trivial Lift [cite: 68]
            s_prev: Previous lifted state (Batch, Num_V, d_model, d_state, H, W)
            
        Returns:
            [cite_start]y_out: Aggregated output (Batch, d_model, H, W) [cite: 83]
            s_next: Next lifted state (Batch, Num_V, d_model, d_state, H, W)
        """
        batch_size, _, H, W = u_t.shape
        
        # [cite_start]--- A. Parameter Generation [cite: 51, 52, 54] ---
        # Delta_t(g) = softplus(Phi_Delta(x_t)(g))
        delta = F.softplus(self.conv_delta(u_t) + self.dt_inv_softplus) # (B, D, H, W)
        B_val = self.conv_B(u_t) # (B, N, H, W)
        C_val = self.conv_C(u_t) # (B, N, H, W)

        # [cite_start]--- B. Discretization (Pointwise in g) [cite: 58, 60, 61] ---
        # bar_A = exp(Delta * A)
        A = -torch.exp(self.log_A_real) # (D, N)
        
        # Broadcast A to spatial dims for pointwise multiplication
        # delta: (B, D, H, W) -> (B, D, 1, H, W)
        # A: (D, N) -> (1, D, N, 1, 1)
        delta_bc = delta.unsqueeze(2)
        A_bc = A.view(1, self.d_model, self.d_state, 1, 1)
        
        # Eq (7): bar_A_t(g)
        A_bar = torch.exp(delta_bc * A_bc) # (B, D, N, H, W)
        
        # Eq (7): bar_B_t(g)
        # 1/A * (exp(delta*A) - 1) * delta * B
        # Simplifies to (exp(delta*A) - 1) / A * B
        inv_A = 1.0 / A_bc
        B_val_bc = B_val.unsqueeze(1) # (B, 1, N, H, W) -> Broadcast across D
        B_bar = (A_bar - 1.0) * inv_A * B_val_bc # (B, D, N, H, W)

        # [cite_start]--- C. Flow-Equivariant Scan Loop [cite: 74, 97] ---
        s_next_list = []
        y_list = []
        
        # Pre-broadcast u_t for the update step
        # [cite_start]Trivial Lift: u_t(v, g) depends only on g [cite: 68]
        # u_t: (B, D, H, W) -> (B, D, 1, H, W)
        u_t_bc = u_t.unsqueeze(2)
        
        # Iterate over generators \nu \in V
        for i, (vx, vy) in enumerate(self.v_list):
            # [cite_start]1. Transport Operator [cite: 39, 40]
            # (T_v s)(v, g) := s(v, psi_1(v)^{-1} g)
            # This is the "roll" operation. psi_1(v)^{-1} shifts coordinate g by -v.
            # So the value at g comes from g-v.
            # In code: shifts=(vy, vx) pulls data from (y-vy, x-vx) to (y, x).
            s_prev_v = s_prev[:, i] # (B, D, N, H, W)
            s_transported = torch.roll(s_prev_v, shifts=(vy, vx), dims=(-2, -1))
            
            # [cite_start]2. Recurrent Update [cite: 79] (Eq 10)
            # s_{t+1}(v, g) = bar_A(g) * s_transported + bar_B(g) * u_t(g)
            s_new = A_bar * s_transported + B_bar * u_t_bc
            s_next_list.append(s_new)
            
            # [cite_start]3. Readout [cite: 80] (Eq 11)
            # y_t(v, g) = C_t(g) * s_{t+1}(v, g) + D_t(g) * u_t(g)
            # Note: C_val is (B, N, H, W). s_new is (B, D, N, H, W).
            # Sum over state dimension N.
            C_val_bc = C_val.unsqueeze(1) # (B, 1, N, H, W)
            y_state = torch.sum(s_new * C_val_bc, dim=2) # Sum over N -> (B, D, H, W)
            
            y_skip = u_t * self.D.view(1, self.d_model, 1, 1)
            y_new = y_state + y_skip # (B, D, H, W)
            y_list.append(y_new)
            
        # Stack back to Lifted Tensor format
        s_next = torch.stack(s_next_list, dim=1) # (B, Num_V, D, N, H, W)
        
        # [cite_start]--- D. Pooling (Flow-Invariant Representation) [cite: 83] ---
        # y_t^inv(g) = Pool_{v \in V} y_t(v, g)
        # We use Max Pooling as it selects the best matching velocity channel.
        y_stacked = torch.stack(y_list, dim=1) # (B, Num_V, D, H, W)
        y_out, _ = torch.max(y_stacked, dim=1) # (B, D, H, W)
        
        return y_out, s_next

class StackedFlowMamba(nn.Module):
    def __init__(self, d_model, d_state, v_range, num_layers=3):
        super().__init__()
        # We share the velocity definition across layers
        self.layers = nn.ModuleList([
            FlowEquivariantMambaCell(d_model, d_state, v_range) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm([d_model, 16, 16]) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.num_v = self.layers[0].num_v

    def init_hidden_state(self, batch_size, h, w, device):
        states = []
        for layer in self.layers:
            # [cite_start]Lifted State Initialization: s_0(v, g) <- 0 [cite: 87]
            s = torch.zeros(batch_size, layer.num_v, layer.d_model, layer.d_state, h, w, device=device)
            states.append(s)
        return states

    def forward(self, x, states):
        next_states = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            s_prev = states[i]
            # Run the Flow-Equivariant Scan
            y_out, s_next = layer(current_input, s_prev)
            
            # Residual + Norm
            current_input = self.norms[i](current_input + y_out)
            next_states.append(s_next)
            
        return current_input, next_states

class LatentFlowMambaModel(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 d_model=64, 
                 d_state=8, 
                 v_range=3,  # MUST cover data velocity range (-3, 3)
                 num_layers=3):
        super().__init__()
        self.d_model = d_model
        
        # 1. G-Equivariant Encoder (ResNet)
        self.encoder = DeepEncoder(input_channels, d_model)
        
        # 2. Flow-Equivariant Mamba Core
        self.stacked_mamba = StackedFlowMamba(d_model, d_state, v_range, num_layers)
        
        # 3. G-Equivariant Decoder
        self.decoder = DeepDecoder(d_model, input_channels)

    def forward(self, input_seq, pred_len=10, teacher_forcing_ratio=0.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # Get Latent spatial dims
        with torch.no_grad():
            dummy_z = self.encoder(torch.zeros(1, C, H, W, device=device))
            _, _, h, w = dummy_z.shape
        
        # Initialize Lifted States
        s_list = self.stacked_mamba.init_hidden_state(batch, h, w, device)
        
        # Process Input Sequence
        for t in range(T_in):
            x_t = input_seq[:, t]
            z_t = self.encoder(x_t)
            z_processed, s_list = self.stacked_mamba(z_t, s_list)
            
        # Prediction Loop
        outputs = []
        last_frame_pred = self.decoder(z_processed) # Initial guess based on last state
        
        # In this specific architecture, we need an autoregressive input for prediction.
        # We use the encoder output of the last frame (or predicted frame).
        curr_z = z_processed 
        
        # To make it strictly autoregressive as per Algorithm 1 lines 4-7:
        # If t > T_in, z_t <- Encoder(x_hat_{t-1})
        last_x_hat = last_frame_pred 

        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                inp_t = target_seq[:, t]
            else:
                inp_t = last_x_hat
                
            z_t = self.encoder(inp_t)
            
            # Flow-Mamba Step
            z_processed, s_list = self.stacked_mamba(z_t, s_list)
            
            # Decode
            pred_frame = self.decoder(z_processed)
            outputs.append(pred_frame)
            last_x_hat = pred_frame
            
        return torch.stack(outputs, dim=1) 

# =============================================================================
# 4. TRAINING UTILS (Unchanged)
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, input_frames):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (seq, _) in enumerate(pbar):
        seq = seq.to(device)
        inp = seq[:, :input_frames]
        target = seq[:, input_frames:]
        
        optimizer.zero_grad()
        preds = model(inp, pred_len=target.shape[1], target_seq=target, teacher_forcing_ratio=0.1)
        
        loss = criterion(preds, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
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
# 5. MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    
    parser.add_argument('--d_model', type=int, default=64) 
    parser.add_argument('--d_state', type=int, default=8) 
    # CRITICAL: v_range must be >= data velocity range (3) to strictly satisfy assumptions
    parser.add_argument('--v_range', type=int, default=3, help='Model velocity range. Must cover data range (-3,3)')
    parser.add_argument('--num_layers', type=int, default=3)
    
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data
    full_dataset = MovingMNISTDataset(
        root=args.data_root, 
        image_size=args.image_size, 
        seq_len=args.seq_len,
        velocity_range_x=(-3, 3), # Data Range
        velocity_range_y=(-3, 3)
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Model
    print(f"Initializing Flow-Equivariant Mamba (Strict Form):")
    
    model = LatentFlowMambaModel(
        input_channels=1,
        d_model=args.d_model,
        d_state=args.d_state,
        v_range=args.v_range, # Ensures V covers G actions in data
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 4. Training Loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.input_frames)
        val_loss = evaluate(model, val_loader, criterion, device, args.input_frames)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{args.epochs} | Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "flow_equivariant_mamba.pth")

if __name__ == "__main__":
    main()