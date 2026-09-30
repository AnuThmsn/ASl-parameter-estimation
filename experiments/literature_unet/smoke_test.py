import torch
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
