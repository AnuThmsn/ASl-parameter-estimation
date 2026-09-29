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

mod_02 = load_module('model_02', 'architectures/02_hierarchical_kinetic_conditioning/model.py')
mod_03 = load_module('model_03', 'architectures/03_gradient_isolated_kinetic_conditioning/model.py')
mod_04 = load_module('model_04', 'architectures/04_partial_gradient_kinetic_conditioning/model.py')

models_to_test = [
    ("Architecture 02 (100%)", mod_02.HierarchicalKineticNet),
    ("Architecture 04 (50%)", mod_04.PartialGradientKineticNet),
    ("Architecture 03 (0%)", mod_03.GradientIsolatedKineticNet)
]

print("Parameter Count Check:")
for name, cls in models_to_test:
    model = cls()
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:25s}: {p_count} params")

print("\n--- Gradient Coupling Test ---")

# We must use the same weight initialization for a fair gradient comparison.
# Let's seed torch before each model creation.
cbf_kinetic_norms = []

for name, cls in models_to_test:
    torch.manual_seed(42)
    model = cls()
    x = torch.randn(1, 4, requires_grad=True)
    
    # CBF Loss backward
    pred, z_kin = model(x)
    cbf_pred = pred[0, 0]
    
    model.zero_grad()
    cbf_pred.backward()
    
    kin_grad_p = next(model.kinetic_encoder.parameters()).grad
    kin_grad_norm = kin_grad_p.norm().item() if kin_grad_p is not None else 0.0
    cbf_kinetic_norms.append((name, kin_grad_norm))
    print(f"{name:25s} CBF -> kinetic_encoder grad norm: {kin_grad_norm:.6f}")

print("\nRelative Gradient Norms (vs Arch02):")
arch02_norm = cbf_kinetic_norms[0][1]
for name, norm in cbf_kinetic_norms:
    rel = norm / arch02_norm if arch02_norm > 0 else 0
    print(f"{name:25s}: {rel*100:.1f}%")
