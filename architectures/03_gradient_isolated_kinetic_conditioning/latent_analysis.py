import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.physics.noise_models import add_noise
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_02 = load_module('model_02', 'architectures/02_hierarchical_kinetic_conditioning/model.py')
mod_03 = load_module('model_03', 'architectures/03_gradient_isolated_kinetic_conditioning/model.py')

HierarchicalKineticNet = mod_02.HierarchicalKineticNet
GradientIsolatedKineticNet = mod_03.GradientIsolatedKineticNet

SEEDS = [42, 123, 2024]
N_TEST = 5_000
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]

def get_latent_data(model, X_ts_norm):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_ts_norm).float()
        h = model.shared_encoder(x)
        z_kin = model.kinetic_encoder(h)
    return z_kin.numpy()

results = []

for seed in SEEDS:
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(20000, rng_norm, n_noise_levels=0)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)

    rng_test = np.random.default_rng(7)
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true = compute_signals_vec(cbf_test, att_test) * SCALE
    X_ts = add_noise(S_true, 0.0, rng=np.random.default_rng(99))
    X_ts_norm = apply_normalization(X_ts, X_mean, X_std)

    models = {
        'Architecture02': HierarchicalKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX),
        'Architecture03': GradientIsolatedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX)
    }
    
    for name, model in models.items():
        if name == 'Architecture02':
            ckpt = f'architectures/02_hierarchical_kinetic_conditioning/checkpoints/Architecture02_seed{seed}.pt'
        else:
            ckpt = f'architectures/03_gradient_isolated_kinetic_conditioning/checkpoints/Architecture03_seed{seed}.pt'
        
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        z_kin = get_latent_data(model, X_ts_norm)
        
        cbf_corrs = [abs(np.corrcoef(z_kin[:, i], cbf_test)[0, 1]) for i in range(z_kin.shape[1])]
        att_corrs = [abs(np.corrcoef(z_kin[:, i], att_test)[0, 1]) for i in range(z_kin.shape[1])]
        
        max_corr_cbf = np.max(cbf_corrs)
        max_corr_att = np.max(att_corrs)
        
        reg_cbf = LinearRegression().fit(z_kin, cbf_test)
        r2_cbf = reg_cbf.score(z_kin, cbf_test)
        
        reg_att = LinearRegression().fit(z_kin, att_test)
        r2_att = reg_att.score(z_kin, att_test)
        
        results.append({
            'Model': name, 'Seed': seed,
            'max_corr_CBF': max_corr_cbf, 'max_corr_ATT': max_corr_att,
            'R2_probe_CBF': r2_cbf, 'R2_probe_ATT': r2_att
        })

df = pd.DataFrame(results)
print("\n=== Latent Representation Analysis ===")
print(df.groupby('Model').mean().drop(columns=['Seed']).to_string())
df.to_csv("architectures/03_gradient_isolated_kinetic_conditioning/tables/latent_purity.csv", index=False)
