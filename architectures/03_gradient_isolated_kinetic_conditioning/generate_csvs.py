import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.physics.noise_models import add_noise
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization
from src.training.evaluation import calc_metrics

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_03 = load_module('model_03', 'architectures/03_gradient_isolated_kinetic_conditioning/model.py')
GradientIsolatedKineticNet = mod_03.GradientIsolatedKineticNet

SEEDS = [42, 123, 2024]
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]
SNRS = [np.inf, 50, 20, 15, 10, 5]
N_TRAIN = 20_000
N_TEST  = 5_000

results = []

for seed in SEEDS:
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm, n_noise_levels=100, sd_max=334.2039/5.0)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)

    rng_test = np.random.default_rng(7)
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true   = compute_signals_vec(cbf_test, att_test) * SCALE
    Y_ts     = np.stack([cbf_test, att_test], axis=1).astype(np.float32)

    model = GradientIsolatedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX)
    ckpt = f"architectures/03_gradient_isolated_kinetic_conditioning/checkpoints/Architecture03_seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    for snr in SNRS:
        rng_noise = np.random.default_rng(99)
        noise_sd = 0.0 if np.isinf(snr) else 334.2039 / snr
        X_ts = add_noise(S_true, noise_sd, rng=rng_noise)
        X_ts_norm = apply_normalization(X_ts, X_mean, X_std)

        phys_pred = model.predict_physical(torch.tensor(X_ts_norm).float()).numpy()
        cbf_m = calc_metrics(Y_ts[:, 0], phys_pred[:, 0])
        att_m = calc_metrics(Y_ts[:, 1], phys_pred[:, 1])
        results.append({
            'Seed': seed, 'Model': 'Architecture03', 'SNR': snr,
            'CBF RMSE': cbf_m['RMSE'], 'CBF MAE': cbf_m['MAE'],
            'CBF R2':   cbf_m['R2'],   'CBF CCC': cbf_m['CCC'],
            'ATT RMSE': att_m['RMSE'], 'ATT MAE': att_m['MAE'],
            'ATT R2':   att_m['R2'],   'ATT CCC': att_m['CCC'],
        })

raw_df = pd.DataFrame(results)
raw_df.to_csv('architectures/03_gradient_isolated_kinetic_conditioning/tables/raw_results.csv', index=False)

agg_df = raw_df.groupby(['Model', 'SNR']).mean().drop(columns=['Seed']).reset_index()
agg_df.to_csv('architectures/03_gradient_isolated_kinetic_conditioning/tables/agg_results.csv', index=False)
