import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

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

mod_05 = load_module('model_05', 'architectures/05_adaptive_gradient_kinetic_conditioning/model.py')
AdaptiveGradientKineticNet = mod_05.AdaptiveGradientKineticNet

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

alpha_samples = []
prediction_results = []
latent_results = []

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

    model = AdaptiveGradientKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX)
    ckpt = f"architectures/05_adaptive_gradient_kinetic_conditioning/checkpoints/Architecture05_seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    # Latent extraction
    with torch.no_grad():
        _, z_kin_tr, _ = model(torch.tensor(X_tr_norm).float())
        z_kin_tr = z_kin_tr.numpy()

    for snr in SNRS:
        rng_noise = np.random.default_rng(99)
        noise_sd = 0.0 if np.isinf(snr) else 334.2039 / snr
        X_ts = add_noise(S_true, noise_sd, rng=rng_noise)
        X_ts_norm = apply_normalization(X_ts, X_mean, X_std)
        
        with torch.no_grad():
            x_t = torch.tensor(X_ts_norm).float()
            phys_pred, alpha = model.predict_physical(x_t)
            phys_pred = phys_pred.numpy()
            alpha = alpha.numpy().flatten()
            
            # We also need z_kin for latent analysis (do this for SNR inf)
            if np.isinf(snr):
                _, z_kin_ts, _ = model(x_t)
                z_kin_ts = z_kin_ts.numpy()
                
                cbf_corrs = np.abs([pearsonr(z_kin_ts[:, i], cbf_test)[0] for i in range(z_kin_ts.shape[1])])
                att_corrs = np.abs([pearsonr(z_kin_ts[:, i], att_test)[0] for i in range(z_kin_ts.shape[1])])
                
                p_cbf = LinearRegression().fit(z_kin_tr, Y_tr[:, 0])
                r2_cbf = r2_score(Y_ts[:, 0], p_cbf.predict(z_kin_ts))
                p_att = LinearRegression().fit(z_kin_tr, Y_tr[:, 1])
                r2_att = r2_score(Y_ts[:, 1], p_att.predict(z_kin_ts))
                
                latent_results.append({
                    'Seed': seed,
                    'max_corr_CBF': np.max(cbf_corrs), 'mean_corr_CBF': np.mean(cbf_corrs), 'median_corr_CBF': np.median(cbf_corrs),
                    'max_corr_ATT': np.max(att_corrs), 'mean_corr_ATT': np.mean(att_corrs), 'median_corr_ATT': np.median(att_corrs),
                    'Test R2 CBF': r2_cbf, 'Test R2 ATT': r2_att
                })
        
        err_cbf = phys_pred[:, 0] - cbf_test
        err_att = phys_pred[:, 1] - att_test
        
        # Save sample data
        df_samp = pd.DataFrame({
            'Seed': seed, 'SNR': snr,
            'CBF_true': cbf_test, 'ATT_true': att_test,
            'CBF_pred': phys_pred[:, 0], 'ATT_pred': phys_pred[:, 1],
            'CBF_err': err_cbf, 'ATT_err': err_att,
            'Alpha': alpha
        })
        alpha_samples.append(df_samp)
        
        c_m = calc_metrics(cbf_test, phys_pred[:, 0])
        a_m = calc_metrics(att_test, phys_pred[:, 1])
        prediction_results.append({
            'Seed': seed, 'SNR': snr,
            'CBF RMSE': c_m['RMSE'], 'ATT RMSE': a_m['RMSE'],
            'CBF MAE': c_m['MAE'], 'ATT MAE': a_m['MAE']
        })

df_a = pd.concat(alpha_samples)
df_a.to_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_sample.csv', index=False)

pd.DataFrame(prediction_results).to_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/prediction_results.csv', index=False)
pd.DataFrame(latent_results).to_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/latent_analysis.csv', index=False)

# By ATT Regime
df_a['ATT_bin'] = df_a['ATT_true'].apply(lambda x: get_bin(x, att_edges))
df_att = df_a.groupby(['SNR', 'ATT_bin']).agg(
    mean_alpha=('Alpha', 'mean'), std_alpha=('Alpha', 'std'),
    CBF_RMSE=('CBF_err', lambda x: np.sqrt(np.mean(x**2))),
    ATT_RMSE=('ATT_err', lambda x: np.sqrt(np.mean(x**2)))
).reset_index()
df_att.to_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_att_regime.csv', index=False)

# By SNR
df_snr = df_a.groupby(['SNR']).agg(mean_alpha=('Alpha', 'mean'), std_alpha=('Alpha', 'std')).reset_index()
df_snr.to_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_snr.csv', index=False)

# By CBF Regime
df_a['CBF_bin'] = pd.cut(df_a['CBF_true'], bins=cbf_edges, labels=cbf_labels, include_lowest=True)
df_cbf = df_a.groupby(['SNR', 'CBF_bin']).agg(mean_alpha=('Alpha', 'mean'), std_alpha=('Alpha', 'std')).reset_index()
df_cbf.to_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_cbf_regime.csv', index=False)

print("Evaluation complete for Architecture 05.")
