import os
import sys
import numpy as np
import pandas as pd
import torch
import importlib.util
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def get_bin(val, edges):
    for i in range(len(edges)-1):
        if edges[i] <= val <= edges[i+1]:
            return f"{edges[i]:.1f}-{edges[i+1]:.1f}"
    return "Out"

att_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

corr_results = []
probe_results = []
regime_results = []

for seed in SEEDS:
    # 1. Regenerate train exactly as in training
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm, n_noise_levels=100, sd_max=334.2039/5.0)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    X_tr_norm = apply_normalization(X_tr, X_mean, X_std)

    # 2. Regenerate test exactly as in evaluation
    rng_test = np.random.default_rng(7)
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true = compute_signals_vec(cbf_test, att_test) * SCALE
    Y_ts = np.stack([cbf_test, att_test], axis=1)
    
    # We will use SNR inf for pure latent analysis to avoid noise obscuring the structural correlations
    rng_noise = np.random.default_rng(99)
    X_ts = add_noise(S_true, 0.0, rng=rng_noise) 
    X_ts_norm = apply_normalization(X_ts, X_mean, X_std)

    models = {
        'Arch02': HierarchicalKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX),
        'Arch03': GradientIsolatedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX)
    }
    
    for name, model in models.items():
        if name == 'Arch02':
            ckpt = f'architectures/02_hierarchical_kinetic_conditioning/checkpoints/Architecture02_seed{seed}.pt'
        else:
            ckpt = f'architectures/03_gradient_isolated_kinetic_conditioning/checkpoints/Architecture03_seed{seed}.pt'
        
        if not os.path.exists(ckpt):
            print(f"ERROR: Checkpoint {ckpt} missing!")
            sys.exit(1)
            
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        
        # Extract z_kin
        z_kin_tr = get_latent_data(model, X_tr_norm)
        z_kin_ts = get_latent_data(model, X_ts_norm)
        
        # Correlations (Test Set)
        cbf_corrs = np.abs([np.corrcoef(z_kin_ts[:, i], cbf_test)[0, 1] for i in range(z_kin_ts.shape[1])])
        att_corrs = np.abs([np.corrcoef(z_kin_ts[:, i], att_test)[0, 1] for i in range(z_kin_ts.shape[1])])
        
        corr_results.append({
            'Architecture': name, 'Seed': seed,
            'max_corr_ATT': np.max(att_corrs), 'mean_corr_ATT': np.mean(att_corrs), 'median_corr_ATT': np.median(att_corrs),
            'max_corr_CBF': np.max(cbf_corrs), 'mean_corr_CBF': np.mean(cbf_corrs), 'median_corr_CBF': np.median(cbf_corrs)
        })
        
        # Probe (Train on Train, Test on Test)
        # We predict raw physical units
        probe_cbf = LinearRegression().fit(z_kin_tr, Y_tr[:, 0])
        pred_cbf_ts = probe_cbf.predict(z_kin_ts)
        r2_cbf = r2_score(Y_ts[:, 0], pred_cbf_ts)
        rmse_cbf = np.sqrt(mean_squared_error(Y_ts[:, 0], pred_cbf_ts))
        mae_cbf = mean_absolute_error(Y_ts[:, 0], pred_cbf_ts)
        
        probe_att = LinearRegression().fit(z_kin_tr, Y_tr[:, 1])
        pred_att_ts = probe_att.predict(z_kin_ts)
        r2_att = r2_score(Y_ts[:, 1], pred_att_ts)
        rmse_att = np.sqrt(mean_squared_error(Y_ts[:, 1], pred_att_ts))
        mae_att = mean_absolute_error(Y_ts[:, 1], pred_att_ts)
        
        probe_results.append({
            'Architecture': name, 'Seed': seed,
            'Test R2 ATT': r2_att, 'Test RMSE ATT': rmse_att, 'Test MAE ATT': mae_att,
            'Test R2 CBF': r2_cbf, 'Test RMSE CBF': rmse_cbf, 'Test MAE CBF': mae_cbf
        })
        
        # Regime analysis
        df_ts = pd.DataFrame({'ATT_true': Y_ts[:, 1], 'CBF_true': Y_ts[:, 0], 'ATT_pred': pred_att_ts, 'CBF_pred': pred_cbf_ts})
        df_ts['ATT_bin'] = df_ts['ATT_true'].apply(lambda x: get_bin(x, att_edges))
        
        for b in df_ts['ATT_bin'].unique():
            sub = df_ts[df_ts['ATT_bin'] == b]
            if len(sub) > 0:
                sub_r2_att = r2_score(sub['ATT_true'], sub['ATT_pred']) if len(sub)>1 else 0
                sub_r2_cbf = r2_score(sub['CBF_true'], sub['CBF_pred']) if len(sub)>1 else 0
                sub_rmse_att = np.sqrt(mean_squared_error(sub['ATT_true'], sub['ATT_pred']))
                sub_rmse_cbf = np.sqrt(mean_squared_error(sub['CBF_true'], sub['CBF_pred']))
                
                regime_results.append({
                    'Architecture': name, 'Seed': seed, 'ATT_bin': b,
                    'Probe RMSE CBF': sub_rmse_cbf, 'Probe RMSE ATT': sub_rmse_att,
                    'Probe R2 CBF': sub_r2_cbf, 'Probe R2 ATT': sub_r2_att
                })

# Save and aggregate
df_corr = pd.DataFrame(corr_results)
df_probe = pd.DataFrame(probe_results)
df_regime = pd.DataFrame(regime_results)

os.makedirs('architectures/03_gradient_isolated_kinetic_conditioning/tables', exist_ok=True)
df_corr.to_csv('architectures/03_gradient_isolated_kinetic_conditioning/tables/latent_correlations.csv', index=False)
df_probe.to_csv('architectures/03_gradient_isolated_kinetic_conditioning/tables/latent_probe.csv', index=False)
df_regime.to_csv('architectures/03_gradient_isolated_kinetic_conditioning/tables/latent_regime.csv', index=False)

print("\n--- Correlations ---")
print(df_corr.groupby('Architecture').mean().drop(columns=['Seed']).to_string())
print("\n--- Probes ---")
print(df_probe.groupby('Architecture').mean().drop(columns=['Seed']).to_string())
print("\n--- Regime (Arch02 vs Arch03 on 0.5-1.0s) ---")
print(df_regime[df_regime['ATT_bin'] == '0.5-1.0'].groupby('Architecture').mean().drop(columns=['Seed']).to_string())
