import sys
import os
import torch
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_05 = load_module('model_05', 'architectures/05_adaptive_gradient_kinetic_conditioning/model.py')
AdaptiveGradientKineticNet = mod_05.AdaptiveGradientKineticNet

print("Parameter Count Check:")
model = AdaptiveGradientKineticNet()
p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Architecture 05: {p_count} params (Expected: 94772 + gate params (90*32+32+32*1+1) = 97685)")

print("\n--- Gradient Coupling Test ---")

# Let's force alpha to specific values by hijacking the gate output
def test_alpha(model, x, force_alpha):
    def hook(module, inputs, outputs):
        return torch.full_like(outputs, force_alpha, requires_grad=True)
    
    # We must patch the sigmoid temporarily or just force the alpha
    # Since we can't easily hook local variables, let's just monkeypatch the forward pass
    
    orig_forward = model.forward
    
    def mock_forward(x_in):
        h = model.shared_encoder(x_in)
        z_kin = model.kinetic_encoder(h)
        att_std = model.att_head(z_kin).squeeze(-1)
        
        # FORCED ALPHA
        alpha = torch.full((x_in.size(0), 1), force_alpha, device=x_in.device, requires_grad=True)
        z_kin_for_cbf = alpha * z_kin + (1.0 - alpha) * z_kin.detach()
        
        cbf_in_tensor = torch.cat([h, z_kin_for_cbf], dim=-1)
        z_cbf = model.cbf_encoder(cbf_in_tensor)
        cbf_std = model.cbf_head(z_cbf).squeeze(-1)
        return torch.stack([cbf_std, att_std], dim=-1), z_kin, alpha

    model.forward = mock_forward
    
    model.zero_grad()
    pred, z_kin, alpha_out = model(x)
    cbf_pred = pred[0, 0]
    cbf_pred.backward()
    
    kin_grad_p = next(model.kinetic_encoder.parameters()).grad
    kin_grad_norm = kin_grad_p.norm().item() if kin_grad_p is not None else 0.0
    
    model.forward = orig_forward
    return kin_grad_norm

torch.manual_seed(42)
m = AdaptiveGradientKineticNet()
x = torch.randn(1, 4, requires_grad=True)

# Get the base 100% gradient
norm_100 = test_alpha(m, x, 1.0)
print(f"Alpha = 1.0 (100%): grad norm = {norm_100:.6f}")

norm_80 = test_alpha(m, x, 0.8)
print(f"Alpha = 0.8 ( 80%): grad norm = {norm_80:.6f} (Expected ~ {norm_100*0.8:.6f})")

norm_50 = test_alpha(m, x, 0.5)
print(f"Alpha = 0.5 ( 50%): grad norm = {norm_50:.6f} (Expected ~ {norm_100*0.5:.6f})")

norm_20 = test_alpha(m, x, 0.2)
print(f"Alpha = 0.2 ( 20%): grad norm = {norm_20:.6f} (Expected ~ {norm_100*0.2:.6f})")

norm_00 = test_alpha(m, x, 0.0)
print(f"Alpha = 0.0 (  0%): grad norm = {norm_00:.6f} (Expected ~ 0.000000)")

