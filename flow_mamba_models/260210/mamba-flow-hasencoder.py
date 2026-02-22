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
        velocity_range_x=(-3, 3), # 稍微增加数据生成的物理速度，以匹配更强的模型
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
# 2. POWERFUL ENCODER / DECODER (ResNet Style)
# =============================================================================

class ResBlock(nn.Module):
    """标准的 ResNet Block"""
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
    """
    更深的 Encoder：
    64x64 -> (Conv+Res) -> 32x32 -> (Conv+Res) -> 16x16
    """
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1: 64 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(32),
            
            # Stage 2: 32 -> 16
            nn.Conv2d(32, d_model, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(d_model),
            ResBlock(d_model) # 多加一层 ResBlock 增强特征
        )
        
    def forward(self, x):
        return self.net(x)

class DeepDecoder(nn.Module):
    """
    更深的 Decoder：
    16x16 -> (Up+Res) -> 32x32 -> (Up+Res) -> 64x64
    """
    def __init__(self, d_model, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1: 16 -> 32
            ResBlock(d_model),
            nn.ConvTranspose2d(d_model, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Stage 2: 32 -> 64
            ResBlock(32),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 3. MODEL: Latent Flow-Equivariant Mamba (Multi-Layer & Large V-Range)
# =============================================================================

class FlowMambaCell(nn.Module):
    def __init__(self, d_model, d_state, v_range, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Reduced Velocity Grid V
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        pad = kernel_size // 2
        
        # Parameter Generators
        self.conv_delta = nn.Conv2d(d_model, d_model, kernel_size, padding=pad, padding_mode='circular', bias=True)
        self.conv_B = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)
        self.conv_C = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)

        # Learnable SSM Parameters
        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_A_real = nn.Parameter(log_A_real)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1))) 

    def forward(self, u_t, s_prev):
        """
        u_t: (B, d_model, h, w)
        s_prev: (B, num_v, d_model, d_state, h, w)
        """
        # --- A. Generate Parameters ---
        delta = F.softplus(self.conv_delta(u_t) + self.dt_inv_softplus)
        B_val = self.conv_B(u_t)
        C_val = self.conv_C(u_t)

        # --- B. Discretization ---
        A = -torch.exp(self.log_A_real)
        
        delta_bc = delta.unsqueeze(2) # (B, D, 1, h, w)
        A_bc = A.view(1, self.d_model, self.d_state, 1, 1)
        
        A_bar = torch.exp(delta_bc * A_bc) # (B, D, N, h, w)
        
        inv_A = 1.0 / A_bc
        B_val_bc = B_val.unsqueeze(1) # (B, 1, N, h, w)
        B_bar = (A_bar - 1.0) * inv_A * B_val_bc # (B, D, N, h, w)

        # --- C. Flow-Equivariant Scan ---
        s_next_list = []
        y_list = []
        
        u_t_bc = u_t.unsqueeze(2) 
        C_val_bc = C_val.unsqueeze(1)
        
        # 注意：这里如果 v_range=5，循环会跑 121 次。
        # 显存是主要瓶颈，计算其实还好。
        for i, (vx, vy) in enumerate(self.v_list):
            # 1. Transport
            s_prev_v = s_prev[:, i]
            s_transported = torch.roll(s_prev_v, shifts=(-vy, -vx), dims=(-2, -1))
            
            # 2. Update
            s_new = A_bar * s_transported + B_bar * u_t_bc
            
            # 3. Readout
            y_state = torch.sum(s_new * C_val_bc, dim=2) 
            y_skip = u_t * self.D.view(1, self.d_model, 1, 1)
            y_new = y_state + y_skip
            
            s_next_list.append(s_new)
            y_list.append(y_new)
            
        s_next = torch.stack(s_next_list, dim=1)
        y_out = torch.stack(y_list, dim=1)       
        
        return y_out, s_next

class StackedFlowMamba(nn.Module):
    """
    堆叠多个 FlowMambaCell，增加深度
    """
    def __init__(self, d_model, d_state, v_range, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            FlowMambaCell(d_model, d_state, v_range) for _ in range(num_layers)
        ])
        # LayerNorm 用于稳定训练
        self.norms = nn.ModuleList([
            nn.LayerNorm([d_model, 16, 16]) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def init_hidden_state(self, batch_size, num_v, h, w, device):
        # 返回一个列表，每个元素对应一层的 hidden state
        states = []
        for _ in range(self.num_layers):
            s = torch.zeros(batch_size, num_v, self.layers[0].d_model, self.layers[0].d_state, h, w, device=device)
            states.append(s)
        return states

    def forward(self, x, states):
        next_states = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            s_prev = states[i]
            
            # 运行 Cell
            # y_out: (B, V, D, h, w)
            y_out, s_next = layer(current_input, s_prev)
            
            # 聚合 V 维度 (Max Pooling 选择最佳速度)
            # 这是连接下一层的关键：下一层不需要知道上一层的具体速度分布，只需要知道"最佳特征"
            z_out, _ = torch.max(y_out, dim=1)
            
            # 残差连接 + Norm
            current_input = self.norms[i](current_input + z_out)
            
            next_states.append(s_next)
            
        return current_input, next_states

class LatentFlowMambaModel(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 d_model=64, 
                 d_state=8, 
                 v_range=2,  # 默认扩大到 5
                 num_layers=3): # 默认堆叠 3 层
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.v_range = v_range
        self.num_v = (2*v_range + 1)**2
        
        # 1. Powerful Encoder
        self.encoder = DeepEncoder(input_channels, d_model)
        
        # 2. Stacked Flow Mamba
        self.stacked_mamba = StackedFlowMamba(d_model, d_state, v_range, num_layers)
        
        # 3. Powerful Decoder
        self.decoder = DeepDecoder(d_model, input_channels)

    def forward(self, input_seq, pred_len=10, teacher_forcing_ratio=0.0, target_seq=None):
        batch, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        
        # Determine latent shape
        with torch.no_grad():
            dummy_z = self.encoder(torch.zeros(1, C, H, W, device=device))
            _, _, h, w = dummy_z.shape
        
        # Initialize Stacked Hidden States
        s_list = self.stacked_mamba.init_hidden_state(batch, self.num_v, h, w, device)
        
        # 1. Process Input Sequence
        for t in range(T_in):
            x_t = input_seq[:, t]
            z_t = self.encoder(x_t)
            
            # Recurrence in Stacked Latent Space
            z_processed, s_list = self.stacked_mamba(z_t, s_list)
            
        # 2. Prediction Loop
        outputs = []
        last_frame = input_seq[:, -1]
        
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                inp_t = target_seq[:, t]
            else:
                inp_t = last_frame
                
            z_t = self.encoder(inp_t)
            
            # Recurrence
            z_processed, s_list = self.stacked_mamba(z_t, s_list)
            
            # Decode
            pred_frame = self.decoder(z_processed)
            outputs.append(pred_frame)
            
            last_frame = pred_frame
            
        return torch.stack(outputs, dim=1) 

# =============================================================================
# 4. TRAINING UTILS
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
        
        # 梯度裁剪很重要，防止多层RNN梯度爆炸
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
    # 显存警告：v_range=5 且多层堆叠非常消耗显存。
    # 如果 OOM (显存不足)，请减小 batch_size 或 d_model
    parser.add_argument('--batch_size', type=int, default=4) # 从 32 降到 16 以安全运行
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    
    # Model Configs
    parser.add_argument('--d_model', type=int, default=64) 
    parser.add_argument('--d_state', type=int, default=8) # 保持较小以节省显存
    parser.add_argument('--v_range', type=int, default=2, help='Range 5 -> 11x11=121 directions')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of stacked Mamba layers')
    
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data
    full_dataset = MovingMNISTDataset(
        root=args.data_root, 
        image_size=args.image_size, 
        seq_len=args.seq_len,
        velocity_range_x=(-3, 3),
        velocity_range_y=(-3, 3)
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Model
    num_directions = (2*args.v_range + 1)**2
    print(f"Initializing Stacked Latent FlowMamba:")
    print(f" - Layers: {args.num_layers}")
    print(f" - V-Range: {args.v_range} (Total directions: {num_directions})")
    print(f" - Encoder: Deep ResNet")
    
    model = LatentFlowMambaModel(
        input_channels=1,
        d_model=args.d_model,
        d_state=args.d_state,
        v_range=args.v_range,
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
            torch.save(model.state_dict(), "stacked_flow_mamba_v5.pth")

if __name__ == "__main__":
    main()