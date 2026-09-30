import sys, os, numpy as np, torch, yaml
sys.path.append('.')
from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import generate_spatial_volume
from src.physics.noise_models import add_noise

with open('experiments/literature_unet/configs/unet_config.yaml') as f:
    cfg = yaml.safe_load(f)
D,H,W = cfg['data']['D'], cfg['data']['H'], cfg['data']['W']

rng_te = np.random.default_rng(7)
sig, cbf, att, mask = generate_spatial_volume(rng_te, D, H, W, noise_sd=0.0)
print(f'Clean signal from generator: mean={sig.mean():.2f}')

nd = np.load('experiments/literature_unet/checkpoints/unet_norm_seed42.npz')
print(f'Train sig_mean: {nd["sig_mean"].flatten()}')
print(f'Train sig_std:  {nd["sig_std"].flatten()}')

sig_norm = (sig - nd['sig_mean']) / nd['sig_std']
print(f'Normalized clean: mean={sig_norm.mean():.3f} std={sig_norm.std():.3f}')

rng_n = np.random.default_rng(99)
flat = sig.transpose(1,2,3,0).reshape(-1,4)
noisy_flat = add_noise(flat, 334.2039/10, rng=rng_n)
noisy = noisy_flat.reshape(D,H,W,4).transpose(3,0,1,2)
noisy_norm = (noisy - nd['sig_mean']) / nd['sig_std']
print(f'Normalized noisy (SNR=10): mean={noisy_norm.mean():.3f} std={noisy_norm.std():.3f}')
