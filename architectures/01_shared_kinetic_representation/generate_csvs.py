import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization
from src.physics.noise_models import add_noise
from src.physics.asl_forward import compute_signals_vec, SCALE
from src.training.evaluation import calc_metrics
from model import SharedKineticNet
from train import BaselineComboNet, AblationIndependentNet

seeds = [42, 123, 2024]
results = []
y_min, y_max = [0.0, 0.5], [100.0, 3.0]

for seed in seeds:
    rng = np.random.default_rng(seed)
    
    # Reload proper normalizers matching training
    X_tr, Y_tr, _ = generate_data(20_000, rng)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)

    models = {
        "Baseline": BaselineComboNet(Y_mean, Y_std, y_min, y_max),
        "Architecture01": SharedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=y_min, y_max=y_max),
        "Ablation": AblationIndependentNet(Y_mean, Y_std, y_min, y_max)
    }
    
    rng_test = np.random.default_rng(7)
    X_ts_clean, Y_ts, _ = generate_data(5_000, rng_test, n_noise_levels=0)
    
    for name, model in models.items():
        model.load_state_dict(torch.load(f"architectures/01_shared_kinetic_representation/checkpoints/{name}_seed{seed}.pt", weights_only=True))
        model.eval()
        
        for snr in [np.inf, 50, 20, 15, 10, 5]:
            if np.isinf(snr):
                X_ts = X_ts_clean
            else:
                sd = 334.2039 / snr
                X_ts = add_noise(X_ts_clean, sd, rng=np.random.default_rng(99))
                
            X_ts_norm = apply_normalization(X_ts, X_mean, X_std)
            phys_pred = model.predict_physical(torch.tensor(X_ts_norm).float()).numpy()
            
            cbf_metrics = calc_metrics(Y_ts[:, 0], phys_pred[:, 0])
            att_metrics = calc_metrics(Y_ts[:, 1], phys_pred[:, 1])
            
            results.append({
                'Seed': seed,
                'Model': name,
                'SNR': snr,
                'CBF RMSE': cbf_metrics['RMSE'],
                'CBF MAE': cbf_metrics['MAE'],
                'CBF R2': cbf_metrics['R2'],
                'CBF CCC': cbf_metrics['CCC'],
                'ATT RMSE': att_metrics['RMSE'],
                'ATT MAE': att_metrics['MAE'],
                'ATT R2': att_metrics['R2'],
                'ATT CCC': att_metrics['CCC']
            })

pd.DataFrame(results).to_csv("architectures/01_shared_kinetic_representation/tables/raw_results.csv", index=False)
df = pd.DataFrame(results)
agg_df = df.groupby(['Model', 'SNR']).mean().drop(columns=['Seed']).reset_index()
agg_df.to_csv("comparisons/architecture_00_vs_01.csv", index=False)
