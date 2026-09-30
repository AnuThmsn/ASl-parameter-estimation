import sys
import os
import torch
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
mod_fixed = importlib.import_module("experiments.gradient_coupling_study.model_fixed")

print("--- GRADIENT COUPLING VERIFICATION ---")
alphas_to_test = [0.00, 0.25, 0.50, 0.75, 1.00]
grad_norms = []

torch.manual_seed(42)
x = torch.randn(1, 4, requires_grad=True)

# To ensure exactly same weights, we create one model and manually override self.alpha
base_model = mod_fixed.FixedGradientKineticNet(alpha=1.0)
base_weights = {k: v.clone() for k, v in base_model.state_dict().items()}

for alpha in alphas_to_test:
    model = mod_fixed.FixedGradientKineticNet(alpha=alpha)
    model.load_state_dict(base_weights)
    
    # Forward check
    with torch.no_grad():
        pred, _ = model(x)
        if alpha == 0.0:
            pred_base = pred
        else:
            diff = torch.abs(pred - pred_base).max().item()
            assert diff < 1e-6, f"Forward pass changed! Diff: {diff}"
            
    # Backward check
    model.zero_grad()
    pred, _ = model(x)
    cbf_pred = pred[0, 0]
    cbf_pred.backward()
    
    kin_grad_p = next(model.kinetic_encoder.parameters()).grad
    kin_grad_norm = kin_grad_p.norm().item() if kin_grad_p is not None else 0.0
    grad_norms.append((alpha, kin_grad_norm))

base_norm = grad_norms[-1][1] # alpha=1.00
for alpha, norm in grad_norms:
    ratio = norm / base_norm if base_norm > 0 else 0
    print(f"alpha={alpha:.2f} gradient_ratio={ratio:.3f} (norm: {norm:.6f})")
    
