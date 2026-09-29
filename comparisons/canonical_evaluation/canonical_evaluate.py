import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
mod_04 = load_module('model_04', 'architectures/04_partial_gradient_kinetic_conditioning/model.py')

HierarchicalKineticNet = mod_02.HierarchicalKineticNet
GradientIsolatedKineticNet = mod_03.GradientIsolatedKineticNet
PartialGradientKineticNet = mod_04.PartialGradientKineticNet

SEEDS = [42, 123, 2024]
N_TRAIN = 20_000
N_TEST = 5_000
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]
SNRS = [np.inf, 50, 20, 15, 10, 5]

att_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
cbf_edges = [0.0, 33.3, 66.6, 100.0]
cbf_labels = ["0-33", "33-66", "66-100"]

def get_bin(val, edges):
    for i in range(len(edges)-1):
        if edges[i] <= val <= edges[i+1]:
            return f"{edges[i]:.1f}-{edges[i+1]:.1f}"
    return "Out"

def calc_metrics(y_true, y_pred):
    if len(y_true) < 2: return {"RMSE": 0, "MAE": 0, "Bias": 0, "r": 0}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    r, _ = pearsonr(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "Bias": bias, "r": r}

overall_results = []
regime_att_results = []
regime_cbf_results = []
latent_results = []

def get_latent_data(model, X_norm):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_norm).float()
        h = model.shared_encoder(x)
        z_kin = model.kinetic_encoder(h)
    return z_kin.numpy()

for seed in SEEDS:
    # 1. Exact Noisy Normalizer
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm, n_noise_levels=100, sd_max=334.2039/5.0)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    X_tr_norm = apply_normalization(X_tr, X_mean, X_std)

    # 2. Canonical Test Set
    rng_test = np.random.default_rng(7)
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true = compute_signals_vec(cbf_test, att_test) * SCALE
    
    Y_ts = np.stack([cbf_test, att_test], axis=1)

    models = {
        'Arch02': HierarchicalKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX),
        'Arch03': GradientIsolatedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX),
        'Arch04': PartialGradientKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX, alpha=0.5)
    }

    # Load checkpoints
    for name, m in models.items():
        if name == 'Arch02': p = f"architectures/02_hierarchical_kinetic_conditioning/checkpoints/Architecture02_seed{seed}.pt"
        elif name == 'Arch03': p = f"architectures/03_gradient_isolated_kinetic_conditioning/checkpoints/Architecture03_seed{seed}.pt"
        else: p = f"architectures/04_partial_gradient_kinetic_conditioning/checkpoints/Architecture04_seed{seed}.pt"
        
        if not os.path.exists(p):
            print(f"CHECKPOINT MISSING: {name}, seed: {seed}")
            sys.exit(1)
        m.load_state_dict(torch.load(p, weights_only=True))
        m.eval()
        
    # Latent Analysis (SNR inf)
    X_ts_inf = add_noise(S_true, 0.0, rng=np.random.default_rng(99))
    X_ts_norm_inf = apply_normalization(X_ts_inf, X_mean, X_std)
    
    for name, m in models.items():
        z_kin_tr = get_latent_data(m, X_tr_norm)
        z_kin_ts = get_latent_data(m, X_ts_norm_inf)
        
        cbf_corrs = np.abs([pearsonr(z_kin_ts[:, i], cbf_test)[0] for i in range(z_kin_ts.shape[1])])
        att_corrs = np.abs([pearsonr(z_kin_ts[:, i], att_test)[0] for i in range(z_kin_ts.shape[1])])
        
        p_cbf = LinearRegression().fit(z_kin_tr, Y_tr[:, 0])
        r2_cbf = r2_score(Y_ts[:, 0], p_cbf.predict(z_kin_ts))
        p_att = LinearRegression().fit(z_kin_tr, Y_tr[:, 1])
        r2_att = r2_score(Y_ts[:, 1], p_att.predict(z_kin_ts))
        
        latent_results.append({
            'Model': name, 'Seed': seed,
            'max_corr_CBF': np.max(cbf_corrs), 'mean_corr_CBF': np.mean(cbf_corrs), 'median_corr_CBF': np.median(cbf_corrs),
            'max_corr_ATT': np.max(att_corrs), 'mean_corr_ATT': np.mean(att_corrs), 'median_corr_ATT': np.median(att_corrs),
            'CBF R2': r2_cbf, 'ATT R2': r2_att
        })

    # Prediction Evaluation across SNRs
    for snr in SNRS:
        rng_noise = np.random.default_rng(99)
        noise_sd = 0.0 if np.isinf(snr) else 334.2039 / snr
        X_ts = add_noise(S_true, noise_sd, rng=rng_noise)
        X_ts_norm = apply_normalization(X_ts, X_mean, X_std)
        X_ts_tensor = torch.tensor(X_ts_norm).float()
        
        for name, m in models.items():
            phys_pred = m.predict_physical(X_ts_tensor).numpy()
            
            # Overall
            cbf_m = calc_metrics(cbf_test, phys_pred[:, 0])
            att_m = calc_metrics(att_test, phys_pred[:, 1])
            overall_results.append({
                'Model': name, 'Seed': seed, 'SNR': snr,
                'CBF RMSE': cbf_m['RMSE'], 'ATT RMSE': att_m['RMSE'],
                'CBF MAE': cbf_m['MAE'], 'ATT MAE': att_m['MAE'],
                'CBF Bias': cbf_m['Bias'], 'ATT Bias': att_m['Bias'],
                'CBF r': cbf_m['r'], 'ATT r': att_m['r']
            })
            
            # Regime ATT
            df_att = pd.DataFrame({'CBF_true': cbf_test, 'ATT_true': att_test, 'CBF_pred': phys_pred[:, 0], 'ATT_pred': phys_pred[:, 1]})
            df_att['ATT_bin'] = df_att['ATT_true'].apply(lambda x: get_bin(x, att_edges))
            for b in df_att['ATT_bin'].unique():
                sub = df_att[df_att['ATT_bin'] == b]
                c_m = calc_metrics(sub['CBF_true'].values, sub['CBF_pred'].values)
                a_m = calc_metrics(sub['ATT_true'].values, sub['ATT_pred'].values)
                regime_att_results.append({
                    'Model': name, 'Seed': seed, 'SNR': snr, 'ATT_bin': b,
                    'CBF RMSE': c_m['RMSE'], 'ATT RMSE': a_m['RMSE']
                })
                
            # Regime CBF
            df_att['CBF_bin'] = pd.cut(df_att['CBF_true'], bins=cbf_edges, labels=cbf_labels, include_lowest=True)
            for b in df_att['CBF_bin'].unique():
                if pd.isna(b): continue
                sub = df_att[df_att['CBF_bin'] == b]
                c_m = calc_metrics(sub['CBF_true'].values, sub['CBF_pred'].values)
                a_m = calc_metrics(sub['ATT_true'].values, sub['ATT_pred'].values)
                regime_cbf_results.append({
                    'Model': name, 'Seed': seed, 'SNR': snr, 'CBF_bin': b,
                    'CBF RMSE': c_m['RMSE'], 'ATT RMSE': a_m['RMSE']
                })

pd.DataFrame(overall_results).to_csv('comparisons/canonical_evaluation/canonical_overall.csv', index=False)
pd.DataFrame(regime_att_results).to_csv('comparisons/canonical_evaluation/canonical_att_regime.csv', index=False)
pd.DataFrame(regime_cbf_results).to_csv('comparisons/canonical_evaluation/canonical_cbf_regime.csv', index=False)
pd.DataFrame(latent_results).to_csv('comparisons/canonical_evaluation/canonical_latent.csv', index=False)
print("Canonical evaluation complete.")
