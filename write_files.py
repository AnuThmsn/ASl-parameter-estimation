import os

base_dir = "experiments/literature_unet"
os.makedirs(f"{base_dir}/configs", exist_ok=True)
os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
os.makedirs(f"{base_dir}/results", exist_ok=True)
os.makedirs(f"{base_dir}/figures", exist_ok=True)

with open(f"{base_dir}/model.py", "w") as f:
    f.write('''import torch
import torch.nn as nn

class LuciwUNet3D(nn.Module):
    """
    Inspired by Luciw et al. 2022 (DOI:10.1002/mrm.29193), adapted to 4 simulated PLDs and synthetic ASL data.
    Input: [B, 4, D, H, W]
    Output: [B, 2, D, H, W] (CBF + ATT)
    """
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.bottleneck = conv_block(base_channels * 4, base_channels * 8)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = conv_block(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        return self.final_conv(d1)

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {count}")
    return count
''')

with open(f"{base_dir}/dataset.py", "w") as f:
    f.write('''import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, PLDs
from src.physics.noise_models import add_noise

def generate_spatial_volume(rng, D=16, H=64, W=64, noise_sd=0.0):
    z, y, x = np.ogrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2
    rz, ry, rx = max(1, D // 2 - 1), max(1, H // 2 - 4), max(1, W // 2 - 4)
    mask = ((z - cz)**2 / rz**2 + (y - cy)**2 / ry**2 + (x - cx)**2 / rx**2) <= 1.0
    
    cbf_map = np.zeros((D, H, W))
    n_blobs_cbf = rng.integers(3, 6)
    for _ in range(n_blobs_cbf):
        bz, by, bx = rng.integers(0, D), rng.integers(0, H), rng.integers(0, W)
        sigma = rng.uniform(2.0, 5.0)
        blob = np.exp(-((z - bz)**2 + (y - by)**2 + (x - bx)**2) / (2 * sigma**2))
        cbf_map += blob * rng.uniform(20.0, 50.0)
    cbf_map += 20.0
    cbf_map = np.clip(cbf_map, 20.0, 90.0)
    cbf_map[~mask] = 0.0
    
    att_map = np.zeros((D, H, W))
    n_blobs_att = rng.integers(3, 6)
    for _ in range(n_blobs_att):
        bz, by, bx = rng.integers(0, D), rng.integers(0, H), rng.integers(0, W)
        sigma = rng.uniform(2.0, 5.0)
        blob = np.exp(-((z - bz)**2 + (y - by)**2 + (x - bx)**2) / (2 * sigma**2))
        att_map += blob * rng.uniform(0.5, 1.5)
    att_map += 1.0
    att_map = np.clip(att_map, 0.5, 3.0)
    att_map[~mask] = 0.0
    
    signals_4d = np.zeros((4, D, H, W))
    mask_indices = np.where(mask)
    if len(mask_indices[0]) > 0:
        cbf_vec = cbf_map[mask_indices]
        att_vec = att_map[mask_indices]
        
        sigs = compute_signals_vec(cbf_vec, att_vec) * SCALE
        
        if noise_sd > 0:
            sigs = add_noise(sigs, noise_sd, rng)
            
        for c in range(4):
            signals_4d[c][mask_indices] = sigs[:, c]
            
    return signals_4d, cbf_map, att_map, mask

class SpatialASLDataset(Dataset):
    def __init__(self, n_volumes, seed, noise_sd=0.0, D=16, H=64, W=64):
        self.n_volumes = n_volumes
        self.noise_sd = noise_sd
        self.rng = np.random.default_rng(seed)
        
        self.signals = []
        self.cbfs = []
        self.atts = []
        self.masks = []
        
        for _ in range(n_volumes):
            sig, cbf, att, mask = generate_spatial_volume(self.rng, D, H, W, noise_sd)
            self.signals.append(sig)
            self.cbfs.append(cbf[np.newaxis, ...])
            self.atts.append(att[np.newaxis, ...])
            self.masks.append(mask[np.newaxis, ...])
            
        self.signals = np.array(self.signals)
        self.cbfs = np.array(self.cbfs)
        self.atts = np.array(self.atts)
        self.masks = np.array(self.masks)
        
        self.signal_mean = np.zeros((4, 1, 1, 1))
        self.signal_std = np.ones((4, 1, 1, 1))
        
    def fit_normalization(self):
        self.signal_mean = self.signals.mean(axis=(0, 2, 3, 4), keepdims=True)[0]
        self.signal_std = self.signals.std(axis=(0, 2, 3, 4), keepdims=True)[0]
        self.signal_std[self.signal_std == 0] = 1.0
        
    def set_normalization(self, mean, std):
        self.signal_mean = mean
        self.signal_std = std
        
    def normalize_signals(self, x):
        return (x - self.signal_mean) / self.signal_std
        
    def __len__(self):
        return self.n_volumes
        
    def __getitem__(self, idx):
        sig = self.normalize_signals(self.signals[idx])
        return (
            torch.tensor(sig, dtype=torch.float32),
            torch.tensor(self.cbfs[idx], dtype=torch.float32),
            torch.tensor(self.atts[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32)
        )
''')

with open(f"{base_dir}/configs/unet_config.yaml", "w") as f:
    f.write('''model:
  in_channels: 4
  base_channels: 16

data:
  D: 16
  H: 64
  W: 64
  n_train: 200
  n_val: 40
  n_test: 40
  train_seed: 42
  val_seed: 100
  test_seed: 7

noise:
  snr_levels: [null, 50, 20, 15, 10, 5]
  ref_signal_scale: 334.2039

training:
  epochs: 200
  batch_size: 2
  lr: 0.0005
  lr_step_epoch: 100
  lr_step_factor: 0.2
  patience: 30
  seeds: [42, 123, 2024]
  grad_clip: 1.0

curriculum:
  phase1_end: 40
  phase2_end: 80
  phase3_end: 120
  phase4_start: 120

loss:
  brain_weight: 1.0
  background_weight: 1.0

cbf_range: [0.0, 100.0]
att_range: [0.5, 3.0]
''')

with open(f"{base_dir}/train.py", "w") as f:
    f.write('''import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset import SpatialASLDataset
from model import LuciwUNet3D

def train():
    with open("experiments/literature_unet/configs/unet_config.yaml") as f:
        config = yaml.safe_load(f)
    pass

if __name__ == "__main__":
    train()
''')

with open(f"{base_dir}/smoke_test.py", "w") as f:
    f.write('''import torch
import torch.nn as nn
import numpy as np
from model import LuciwUNet3D, count_parameters
from dataset import generate_spatial_volume, SpatialASLDataset

def test():
    try:
        rng = np.random.default_rng(0)
        sig, cbf, att, mask = generate_spatial_volume(rng, D=16, H=32, W=32, noise_sd=0.0)
        print(f"Generated volume shapes: sig={sig.shape}, cbf={cbf.shape}, att={att.shape}")
        print(f"Mask active voxels: {mask.sum()}")
        
        model = LuciwUNet3D()
        params = count_parameters(model)
        
        x = torch.randn(1, 4, 16, 32, 32)
        out = model(x)
        print(f"Output shape: {out.shape}")
        assert out.shape == (1, 2, 16, 32, 32), "Output shape mismatch"
        
        target = torch.randn(1, 2, 16, 32, 32)
        mask_t = torch.ones(1, 1, 16, 32, 32)
        
        loss = nn.L1Loss()(out * mask_t, target * mask_t)
        loss.backward()
        
        grad_norm = model.enc1[0].weight.grad.norm().item()
        print(f"Gradients flow: {grad_norm > 0}")
        assert grad_norm > 0, "No gradients"
        
        x_missing = x.clone()
        x_missing[:, 0] = 0.0
        out_missing = model(x_missing)
        assert out_missing.shape == (1, 2, 16, 32, 32), "Missing PLD shape mismatch"
        print("Missing PLD test: PASS")
        
        ds = SpatialASLDataset(5, 0, D=16, H=32, W=32)
        ds.fit_normalization()
        s, c, a, m = ds[0]
        print(f"Normalized signal mean: {s.mean().item():.3f}, std: {s.std().item():.3f}")
        assert abs(s.mean().item()) < 0.5, "Norm mean off"
        
        print("SMOKE TEST: PASS")
        
    except Exception as e:
        print(f"SMOKE TEST: FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
''')
