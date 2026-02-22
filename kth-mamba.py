import argparse
import random
import re
import urllib.request
import zipfile
import shutil
import os
import cv2
import numpy as np
import h5py
import math
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ===========================================================================
# 1. KTH DATASET UTILS (Downloaded & Preprocessing)
# ===========================================================================

_BASE = "https://www.csc.kth.se/cvap/actions/"
_ACTIONS = [
    ("walking",      242),
    ("jogging",      168),
    ("running",      149),
    ("boxing",       194),
    ("handwaving",   218),
    ("handclapping", 176),
]
_ACTION2IDX = {name: idx for idx, (name, _) in enumerate(_ACTIONS)}
_SEQ_TXT = "00sequences.txt"

def _safe_dl(url: str, dst: Path) -> Path:
    """Download *url* to *dst* unless it already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    print(f"Downloading {url} to {dst}...")
    with urllib.request.urlopen(url) as resp, open(dst, "wb") as fh:
        total = int(resp.getheader("Content-Length", "0"))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dst.name)
        while buf := resp.read(16 << 10):
            fh.write(buf)
            pbar.update(len(buf))
        pbar.close()
    return dst

def _download_kth(root: Path) -> Path:
    """Ensure the KTH archives + annotation file are present; return video dir."""
    root = root.expanduser()
    video_dir = root / "kth_actions"
    video_dir.mkdir(parents=True, exist_ok=True)

    # annotation list
    _safe_dl(_BASE + _SEQ_TXT, root / _SEQ_TXT)

    # each action archive
    for act, mib in _ACTIONS:
        zip_path = root / f"{act}.zip"
        if not zip_path.exists():
            _safe_dl(_BASE + f"{act}.zip", zip_path)
        marker = video_dir / f".{act}_unzipped"
        if not marker.exists():
            print(f"Extracting {zip_path.name} ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(video_dir)
            marker.touch()
    return video_dir

def preprocess_videos(video_dir: Path, hdf5_path: Path, height: int = 64, width: int = 64) -> None:
    """Pre-extract frames from all videos into a single HDF5 file."""
    if hdf5_path.exists():
        return
        
    def process_video(video_path: Path) -> Tuple[str, np.ndarray]:
        video_name = video_path.stem
        if video_name.endswith("_uncomp"):
            video_name = video_name[:-7]
            
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        cap.release()
        return video_name, np.stack(frames) if frames else np.zeros((0, height, width), dtype=np.uint8)
    
    # Process all videos in parallel
    video_paths = list(video_dir.glob("*.avi"))
    print(f"Pre-extracting frames from {len(video_paths)} videos to {hdf5_path}...")
    
    with h5py.File(hdf5_path, 'w') as f:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            futures = [executor.submit(process_video, vp) for vp in video_paths]
            for future in tqdm(futures, total=len(video_paths)):
                video_name, frames = future.result()
                if len(frames) > 0:
                    f.create_dataset(video_name, data=frames, compression='gzip', compression_opts=1)

# ===========================================================================
# 2. DATASET CLASS
# ===========================================================================

class KTHVideoClips(Dataset):
    """Return `(clip, label)` where *clip* has shape `(T,1,H,W)` and label in [0,5]."""
    _LINE_RE = re.compile(r"^(?P<fname>\S+)\s+frames\s+(?P<ranges>[\d,\s\-]+)$")

    def __init__(self, root: Path, seq_len: int = 16, height: int = 64, width: int = 64,
                 step: int = 1, split: str = 'train', vx_data_range: int = 0, vy_data_range: int = 0,
                 split_method: str = 'person'):
        self.video_root = _download_kth(root)
        self.hdf5_path = root / "kth_frames.h5"
        self.seq_len, self.step = seq_len, step
        self.split = split
        self.vx_data_range = vx_data_range
        self.vy_data_range = vy_data_range
        
        # Pre-extract if needed
        preprocess_videos(self.video_root, self.hdf5_path, height, width)
        
        # Load dataset
        print(f"Loading {split} dataset from HDF5...")
        self.frames_dict = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for k in f.keys():
                self.frames_dict[k] = f[k][:]

        # Parse sequences
        self.sequences = self._parse_sequences(root / _SEQ_TXT, split_method)
        
        # Generate deterministic velocities for data augmentation (Camera Motion)
        if vx_data_range > 0 or vy_data_range > 0:
            rng = np.random.RandomState(42)
            self.velocities = []
            for _ in range(len(self.sequences)):
                vx = rng.randint(-vx_data_range, vx_data_range + 1)
                vy = rng.randint(-vy_data_range, vy_data_range + 1)
                self.velocities.append((vy, vx))
        else:
            self.velocities = None
            
        print(f"Loaded {len(self.sequences)} sequences for {split}.")

    def _parse_sequences(self, txt_path, split_method):
        all_sequences = []
        with open(txt_path) as fh:
            for line in fh:
                m = self._LINE_RE.match(line.strip())
                if not m: continue
                vname = m.group("fname")
                rng_blob = m.group("ranges")
                if vname not in self.frames_dict: continue
                total_frames = len(self.frames_dict[vname])
                
                for rng in rng_blob.split(','):
                    rng = rng.strip()
                    if not rng: continue
                    if '-' in rng: s, e = map(int, rng.split('-'))
                    else: s = e = int(rng)
                    e = min(e, total_frames - 1)
                    if e - s + 1 >= self.seq_len * self.step:
                        all_sequences.append((vname, s, e))
        
        # Split logic
        if split_method == 'person':
            # 1-16 train, 17-20 val, 21-25 test
            sequences = []
            for vname, s, e in all_sequences:
                pid = int(vname.split('_')[0][6:8])
                if self.split == 'train' and pid <= 16: sequences.append((vname, s, e))
                elif self.split == 'val' and 17 <= pid <= 20: sequences.append((vname, s, e))
                elif self.split == 'test' and pid >= 21: sequences.append((vname, s, e))
            return sequences
        else:
            rng = np.random.RandomState(42)
            rng.shuffle(all_sequences)
            n = len(all_sequences)
            tr, va = int(0.7*n), int(0.15*n)
            if self.split == 'train': return all_sequences[:tr]
            elif self.split == 'val': return all_sequences[tr:tr+va]
            else: return all_sequences[tr+va:]

    def __len__(self): return len(self.sequences)

    def _apply_velocity_shift(self, frame, vy, vx):
        # frame: (1, H, W)
        _, H, W = frame.shape
        vy, vx = vy % H, vx % W
        if vy == 0 and vx == 0: return frame
        
        # We roll using torch.roll for simplicity and efficiency on circular boundary
        return torch.roll(frame, shifts=(vy, vx), dims=(1, 2))

    def __getitem__(self, idx):
        vname, s, e = self.sequences[idx]
        max_start = e - self.seq_len * self.step + 1
        start = random.randint(s, max_start)
        
        frames = []
        for i in range(self.seq_len):
            idx_f = start + i * self.step
            frame = self.frames_dict[vname][idx_f]
            frame = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
            frames.append(frame)
        clip = torch.stack(frames, dim=0) # (T, 1, H, W)

        # Apply camera motion (data augmentation)
        if self.velocities is not None:
            vy, vx = self.velocities[idx]
            shifted = []
            for t, f in enumerate(clip):
                # Motion is cumulative: v * t
                shifted.append(self._apply_velocity_shift(f, vy*t, vx*t))
            clip = torch.stack(shifted, dim=0)

        # Simple Augmentation
        if self.split == 'train' and random.random() < 0.5:
            clip = torch.flip(clip, dims=[3])

        action = vname.split('_')[1]
        label = _ACTION2IDX[action]
        return clip, label

# =============================================================================
# 3. MODEL: FLOW-EQUIVARIANT MAMBA CLASSIFIER
# =============================================================================

class FlowMambaCell(nn.Module):
    """
    Translation Flow-Equivariant Mamba Cell.
    Uses vectorized transport (gather/roll) and stabilized SSM scan.
    """
    def __init__(self, d_model, d_state, v_list, image_size, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.v_list = v_list # List of (vy, vx) tuples
        self.num_v = len(v_list)
        self.image_size = image_size

        # Precompute Gather Indices for Vectorized Transport
        # Shape: (1, num_v, 1, 1, H*W)
        H, W = image_size, image_size
        gather_indices = []
        for (vy, vx) in self.v_list:
            # Source coordinates for inverse flow (roll by -v)
            # y_target = y_source + vy -> y_source = y_target - vy
            y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            src_y = (y_grid - vy) % H
            src_x = (x_grid - vx) % W
            flat_idx = src_y * W + src_x
            gather_indices.append(flat_idx)
        gather_indices = torch.stack(gather_indices, dim=0)
        self.register_buffer('gather_indices', gather_indices.view(1, self.num_v, 1, 1, H*W))

        # Normalization
        self.norm = nn.GroupNorm(4, d_model)

        # Parameter Generators (Selective Scan)
        pad = kernel_size // 2
        self.conv_delta = nn.Conv2d(d_model, d_model, kernel_size, padding=pad, padding_mode='circular', bias=True)
        self.conv_B = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)
        self.conv_C = nn.Conv2d(d_model, d_state, kernel_size, padding=pad, padding_mode='circular', bias=False)

        # Learnable SSM Parameters
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
        V = self.num_v

        u_norm = self.norm(u_t)

        # 1. Generate Parameters
        delta = F.softplus(self.conv_delta(u_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=1e-4, max=5.0)
        B_val = self.conv_B(u_norm)
        C_val = self.conv_C(u_norm)

        # 2. Discretize (Euler)
        A = -torch.exp(self.log_A_real)
        delta_bc = delta.unsqueeze(2) # (B,D,1,H,W)
        A_bc = A.view(1, D, N, 1, 1)
        B_val_bc = B_val.unsqueeze(1) # (B,1,N,H,W)
        
        A_bar = torch.exp(delta_bc * A_bc)
        B_bar = delta_bc * B_val_bc 

        # 3. Vectorized Transport
        s_flat = s_prev.view(B, V, D, N, H*W)
        indices = self.gather_indices.expand(B, V, D, N, H*W)
        s_transported = torch.gather(s_flat, -1, indices).view(B, V, D, N, H, W)

        # 4. Update
        u_bc = u_t.view(B, 1, D, 1, H, W)
        s_new = A_bar.unsqueeze(1) * s_transported + B_bar.unsqueeze(1) * u_bc

        # 5. Readout
        C_bc = C_val.view(B, 1, 1, N, H, W)
        y_state = torch.sum(s_new * C_bc, dim=3) # Sum over N
        y_skip = u_t.unsqueeze(1) * self.D.view(1, 1, D, 1, 1)
        y_out = y_state + y_skip # (B, V, D, H, W)

        return y_out, s_new

class FlowMambaClassifier(nn.Module):
    def __init__(self, input_channels, d_model, d_state, image_size, num_classes, 
                 vx_range=0, vy_range=0):
        super().__init__()
        self.d_model = d_model
        
        # Define V list based on ranges
        self.v_list = [(vy, vx) for vx in range(-vx_range, vx_range + 1) 
                                for vy in range(-vy_range, vy_range + 1)]
        self.num_v = len(self.v_list)
        print(f"Initializing FlowMambaClassifier with |V|={self.num_v} velocities.")

        # Encoder (Lift frame to d_model)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, padding=2, padding_mode='circular'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d_model, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # Recurrent Core
        self.cell = FlowMambaCell(d_model, d_state, self.v_list, image_size)

        # Classifier Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Init Hidden State
        s = torch.zeros(B, self.num_v, self.d_model, self.cell.d_state, H, W, device=x.device)
        
        # Recurrent Loop
        for t in range(T):
            frame = x[:, t]
            u_t = self.encoder(frame)
            y_lifted, s = self.cell(u_t, s)
            
        # Classification on final state
        # 1. Invariance Pooling: Max over V
        # y_lifted: (B, V, D, H, W)
        feat_v, _ = torch.max(y_lifted, dim=1) # (B, D, H, W)
        
        # 2. Spatial Pooling: Avg over H, W
        feat_spatial = self.pool(feat_v).flatten(1) # (B, D)
        
        # 3. Logits
        logits = self.head(feat_spatial)
        return logits

# =============================================================================
# 4. TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for clips, labels in pbar:
        clips, labels = clips.to(device), labels.to(device) # clip: (B, T, 1, H, W)
        
        opt.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        loss_sum += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.3f}"})
        
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for clips, labels in pbar:
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        
        loss_sum += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += clips.size(0)
        
    return loss_sum / total, correct / total

# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/kth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vx_range", type=int, default=0, help="Equivariant range X")
    parser.add_argument("--vy_range", type=int, default=0, help="Equivariant range Y")
    parser.add_argument("--vx_data_range", type=int, default=0, help="Data Augmentation X")
    parser.add_argument("--vy_data_range", type=int, default=0, help="Data Augmentation Y")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset
    print("Initializing Datasets...")
    train_ds = KTHVideoClips(Path(args.data_root), split='train', 
                             vx_data_range=args.vx_data_range, vy_data_range=args.vy_data_range)
    val_ds   = KTHVideoClips(Path(args.data_root), split='val', 
                             vx_data_range=args.vx_data_range, vy_data_range=args.vy_data_range)
    test_ds  = KTHVideoClips(Path(args.data_root), split='test', 
                             vx_data_range=args.vx_data_range, vy_data_range=args.vy_data_range)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model
    print(f"Building FlowMamba Classifier (vx_range={args.vx_range}, vy_range={args.vy_range})...")
    model = FlowMambaClassifier(
        input_channels=1,
        d_model=args.d_model,
        d_state=args.d_state,
        image_size=args.img_size,
        num_classes=6,
        vx_range=args.vx_range,
        vy_range=args.vy_range
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Val")
        
        # Test occasionally or if best val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
            torch.save(model.state_dict(), "kth_best_model.pth")
            marker = "*"
        else:
            test_loss, test_acc = 0.0, 0.0
            marker = ""
            
        print(f"Epoch {epoch} | T: {time.time()-start:.1f}s | "
              f"Tr Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} {marker}")
        
        if marker:
            print(f"    >>> New Best! Test Acc: {test_acc:.3f}")

if __name__ == "__main__":
    main()