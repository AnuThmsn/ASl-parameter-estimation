import sys
import os
import torch
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_02 = load_module('model_02', 'architectures/02_hierarchical_kinetic_conditioning/model.py')
mod_03 = load_module('model_03', 'architectures/03_gradient_isolated_kinetic_conditioning/model.py')

HierarchicalKineticNet = mod_02.HierarchicalKineticNet
GradientIsolatedKineticNet = mod_03.GradientIsolatedKineticNet

def check_gradients(model_class, name):
    print(f"\n--- Testing {name} ---")
    model = model_class()
    x = torch.randn(1, 4, requires_grad=True)
    
    pred, z_kin = model(x)
    cbf_pred = pred[0, 0]
    att_pred = pred[0, 1]
    
    model.zero_grad()
    att_pred.backward(retain_graph=True)
    
    kin_grad_norm_att = next(model.kinetic_encoder.parameters()).grad.norm().item()
    shared_grad_norm_att = next(model.shared_encoder.parameters()).grad.norm().item()
    print(f"ATT loss -> kinetic_encoder grad norm: {kin_grad_norm_att:.4f} (Expected > 0)")
    print(f"ATT loss -> shared_encoder grad norm: {shared_grad_norm_att:.4f} (Expected > 0)")
    
    model.zero_grad()
    cbf_pred.backward(retain_graph=True)
    
    cbf_grad_norm_cbf = next(model.cbf_encoder.parameters()).grad.norm().item()
    kin_grad_cbf_p = next(model.kinetic_encoder.parameters()).grad
    kin_grad_norm_cbf = kin_grad_cbf_p.norm().item() if kin_grad_cbf_p is not None else 0.0
    shared_grad_norm_cbf = next(model.shared_encoder.parameters()).grad.norm().item()
    
    print(f"CBF loss -> cbf_encoder grad norm: {cbf_grad_norm_cbf:.4f} (Expected > 0)")
    print(f"CBF loss -> kinetic_encoder grad norm: {kin_grad_norm_cbf:.4f} " + 
          ("(Expected > 0)" if name == "Architecture 02" else "(Expected == 0)"))
    print(f"CBF loss -> shared_encoder grad norm: {shared_grad_norm_cbf:.4f} (Expected > 0)")

    x = torch.randn(1, 4)
    h = model.shared_encoder(x)
    z_kin_test = model.kinetic_encoder(h)
    
    z_kin_for_cbf = z_kin_test.clone().detach().requires_grad_(True)
    cbf_in_2 = torch.cat([h, z_kin_for_cbf], dim=-1)
    z_cbf_2 = model.cbf_encoder(cbf_in_2)
    cbf_std_2 = model.cbf_head(z_cbf_2).squeeze(-1)
    cbf_std_2.backward()
    print(f"Representation dependence: d(CBF)/d(z_kin_value) norm: {z_kin_for_cbf.grad.norm().item():.4f} (Expected > 0)")


check_gradients(HierarchicalKineticNet, "Architecture 02")
check_gradients(GradientIsolatedKineticNet, "Architecture 03")
