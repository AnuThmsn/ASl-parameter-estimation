import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

mod_04 = load_module('model_04', 'architectures/04_partial_gradient_kinetic_conditioning/model.py')
PartialGradientKineticNet = mod_04.PartialGradientKineticNet

SEEDS = [42, 123, 2024]
N_TRAIN = 20_000
N_TEST = 5_000
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]

def get_latent_data(model, X_norm):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_norm).float()
        h = model.shared_encoder(x)
        z_kin = model.kinetic_encoder(h)
    return z_kin.numpy()

results = []

for seed in SEEDS:
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm, n_noise_levels=100, sd_max=334.2039/5.0)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    X_tr_norm = apply_normalization(X_tr, X_mean, X_std)

    rng_test = np.random.default_rng(7)
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true = compute_signals_vec(cbf_test, att_test) * SCALE
    Y_ts = np.stack([cbf_test, att_test], axis=1)
    
    rng_noise = np.random.default_rng(99)
    X_ts = add_noise(S_true, 0.0, rng=rng_noise)
    X_ts_norm = apply_normalization(X_ts, X_mean, X_std)

    model = PartialGradientKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX, alpha=0.5)
    ckpt = f'architectures/04_partial_gradient_kinetic_conditioning/checkpoints/Architecture04_seed{seed}.pt'
    model.load_state_dict(torch.load(ckpt, weights_only=True))
        
    z_kin_tr = get_latent_data(model, X_tr_norm)
    z_kin_ts = get_latent_data(model, X_ts_norm)
        
    probe_cbf = LinearRegression().fit(z_kin_tr, Y_tr[:, 0])
    pred_cbf_ts = probe_cbf.predict(z_kin_ts)
    r2_cbf = r2_score(Y_ts[:, 0], pred_cbf_ts)
        
    probe_att = LinearRegression().fit(z_kin_tr, Y_tr[:, 1])
    pred_att_ts = probe_att.predict(z_kin_ts)
    r2_att = r2_score(Y_ts[:, 1], pred_att_ts)
        
    results.append({
        'Model': 'Architecture04', 'Seed': seed,
        'Test R2 CBF': r2_cbf, 'Test R2 ATT': r2_att
    })

df = pd.DataFrame(results)
print(df.groupby('Model').mean().drop(columns=['Seed']).to_string())
df.to_csv('architectures/04_partial_gradient_kinetic_conditioning/tables/latent_probe.csv', index=False)
