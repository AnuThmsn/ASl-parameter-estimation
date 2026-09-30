import sys
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
