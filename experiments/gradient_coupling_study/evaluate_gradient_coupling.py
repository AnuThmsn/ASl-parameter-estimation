import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.physics.noise_models import add_noise
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization

# Dynamic loading
mod_fixed = importlib.import_module("experiments.gradient_coupling_study.model_fixed")
mod_adapt = importlib.import_module("architectures.05_adaptive_gradient_kinetic_conditioning.model")
FixedGradientKineticNet = mod_fixed.FixedGradientKineticNet
AdaptiveGradientKineticNet = mod_adapt.AdaptiveGradientKineticNet

SEEDS = [42, 123, 2024]
N_TRAIN = 20_000
N_TEST = 5_000
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]
SNRS = [np.inf, 50, 20, 15, 10, 5]
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

att_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
cbf_edges = [0.0, 33.3, 66.6, 100.0]
cbf_labels = ["0-33", "33-66", "66-100"]

def get_bin(val, edges):
    for i in range(len(edges)-1):
        if edges[i] <= val <= edges[i+1]:
            return f"{edges[i]:.1f}-{edges[i+1]:.1f}"
    return "Out"

def calc_metrics(y_true, y_pred):
    if len(y_true) < 2: return {"RMSE": 0, "MAE": 0}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae}

raw_results = []
att_regime_results = []
cbf_regime_results = []
alpha_samples = []

print("Starting evaluation...")

for seed in SEEDS:
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm, n_noise_levels=100, sd_max=334.2039/5.0)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)

    rng_test = np.random.default_rng(7)
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true = compute_signals_vec(cbf_test, att_test) * SCALE
    
    models = {}
    for a in ALPHAS:
        m = FixedGradientKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX, alpha=a)
        m.load_state_dict(torch.load(f"experiments/gradient_coupling_study/checkpoints/fixed_alpha_{a:.2f}_seed{seed}.pt", weights_only=True))
        m.eval()
        models[f"alpha_{a:.2f}"] = m
        
    m_adapt = AdaptiveGradientKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX)
    m_adapt.load_state_dict(torch.load(f"architectures/05_adaptive_gradient_kinetic_conditioning/checkpoints/Architecture05_seed{seed}.pt", weights_only=True))
    m_adapt.eval()
    models["Adaptive"] = m_adapt
    
    for snr in SNRS:
        rng_noise = np.random.default_rng(99)
        noise_sd = 0.0 if np.isinf(snr) else 334.2039 / snr
        X_ts = add_noise(S_true, noise_sd, rng=rng_noise)
        X_ts_norm = apply_normalization(X_ts, X_mean, X_std)
        X_ts_tensor = torch.tensor(X_ts_norm).float()
        
        for name, m in models.items():
            if name == "Adaptive":
                phys_pred, alpha_pred = m.predict_physical(X_ts_tensor)
                phys_pred = phys_pred.numpy()
                alpha_pred = alpha_pred.numpy().flatten()
                
                df_alpha = pd.DataFrame({
                    'Seed': seed, 'SNR': snr, 'Sample_ID': np.arange(N_TEST),
                    'CBF': cbf_test, 'ATT': att_test, 'Alpha': alpha_pred
                })
                alpha_samples.append(df_alpha)
            else:
                phys_pred = m.predict_physical(X_ts_tensor).numpy()
                
            c_m = calc_metrics(cbf_test, phys_pred[:, 0])
            a_m = calc_metrics(att_test, phys_pred[:, 1])
            raw_results.append({
                'model': name, 'seed': seed, 'snr': snr,
                'cbf_rmse': c_m['RMSE'], 'att_rmse': a_m['RMSE'],
                'cbf_mae': c_m['MAE'], 'att_mae': a_m['MAE']
            })
            
            df_att = pd.DataFrame({'CBF_true': cbf_test, 'ATT_true': att_test, 'CBF_pred': phys_pred[:, 0], 'ATT_pred': phys_pred[:, 1]})
            df_att['ATT_bin'] = df_att['ATT_true'].apply(lambda x: get_bin(x, att_edges))
            for b in df_att['ATT_bin'].unique():
                sub = df_att[df_att['ATT_bin'] == b]
                c_m = calc_metrics(sub['CBF_true'].values, sub['CBF_pred'].values)
                a_m = calc_metrics(sub['ATT_true'].values, sub['ATT_pred'].values)
                att_regime_results.append({
                    'model': name, 'seed': seed, 'snr': snr, 'ATT_bin': b,
                    'cbf_rmse': c_m['RMSE'], 'att_rmse': a_m['RMSE']
                })
                
            df_att['CBF_bin'] = pd.cut(df_att['CBF_true'], bins=cbf_edges, labels=cbf_labels, include_lowest=True)
            for b in df_att['CBF_bin'].unique():
                if pd.isna(b): continue
                sub = df_att[df_att['CBF_bin'] == b]
                c_m = calc_metrics(sub['CBF_true'].values, sub['CBF_pred'].values)
                a_m = calc_metrics(sub['ATT_true'].values, sub['ATT_pred'].values)
                cbf_regime_results.append({
                    'model': name, 'seed': seed, 'snr': snr, 'CBF_bin': b,
                    'cbf_rmse': c_m['RMSE'], 'att_rmse': a_m['RMSE']
                })

print("Saving results...")
df_raw = pd.DataFrame(raw_results)
df_raw.to_csv('experiments/gradient_coupling_study/results/raw_results.csv', index=False)

df_att = pd.DataFrame(att_regime_results)
df_att.to_csv('experiments/gradient_coupling_study/results/att_regime_results.csv', index=False)

df_cbf = pd.DataFrame(cbf_regime_results)
df_cbf.to_csv('experiments/gradient_coupling_study/results/cbf_regime_results.csv', index=False)

df_a = pd.concat(alpha_samples)
df_a.to_csv('experiments/gradient_coupling_study/results/alpha_samples.csv', index=False)

# Summary Results (mean / std)
df_sum = df_raw.groupby(['model', 'snr']).agg(
    cbf_rmse_mean=('cbf_rmse', 'mean'), cbf_rmse_std=('cbf_rmse', 'std'),
    att_rmse_mean=('att_rmse', 'mean'), att_rmse_std=('att_rmse', 'std')
).reset_index()
df_sum.to_csv('experiments/gradient_coupling_study/results/summary_results.csv', index=False)

df_per_seed = df_raw.pivot_table(index=['model', 'seed'], columns='snr', values='cbf_rmse').reset_index()
df_per_seed.to_csv('experiments/gradient_coupling_study/results/per_seed_results.csv', index=False)

print("Evaluation finished.")
