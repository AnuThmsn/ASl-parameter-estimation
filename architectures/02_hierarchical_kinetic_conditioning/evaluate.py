import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.physics.noise_models import add_noise
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization
from src.training.evaluation import calc_metrics
from model import HierarchicalKineticNet, HierarchicalKineticNet_NoCondition

SEEDS = [42, 123, 2024]
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]
N_TRAIN = 20_000
N_TEST  = 5_000

def get_bin(val, edges):
    for i in range(len(edges)-1):
        if edges[i] <= val <= edges[i+1]:
            return f"{edges[i]:.1f}-{edges[i+1]:.1f}"
    return "Out"

att_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
cbf_edges = [0.0, 33.3, 66.6, 100.0]
cbf_labels = ["Low", "Med", "High"]

results_att = []
results_cbf = []

for seed in SEEDS:
    rng_norm = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm, n_noise_levels=0)
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
        'Ablation': HierarchicalKineticNet_NoCondition(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX, cbf_hidden=100),
    }

    for name, model in models.items():
        ckpt = f"architectures/02_hierarchical_kinetic_conditioning/checkpoints/{name}_seed{seed}.pt"
        if name == 'Architecture02':
            ckpt = f"architectures/02_hierarchical_kinetic_conditioning/checkpoints/Architecture02_seed{seed}.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        
        phys_pred = model.predict_physical(torch.tensor(X_ts_norm).float()).numpy()
        err_cbf = (phys_pred[:, 0] - cbf_test)**2
        err_att = (phys_pred[:, 1] - att_test)**2
        
        df = pd.DataFrame({"CBF_true": cbf_test, "ATT_true": att_test, "CBF_err_sq": err_cbf, "ATT_err_sq": err_att})
        
        df["ATT_bin"] = df["ATT_true"].apply(lambda x: get_bin(x, att_edges))
        for b in df["ATT_bin"].unique():
            sub = df[df["ATT_bin"] == b]
            results_att.append({
                "Model": name, "Seed": seed, "ATT_bin": b,
                "CBF_RMSE": np.sqrt(sub["CBF_err_sq"].mean()),
                "ATT_RMSE": np.sqrt(sub["ATT_err_sq"].mean()),
                "Count": len(sub)
            })
            
        df["CBF_bin"] = pd.cut(df["CBF_true"], bins=cbf_edges, labels=cbf_labels, include_lowest=True)
        for b in df["CBF_bin"].unique():
            sub = df[df["CBF_bin"] == b]
            results_cbf.append({
                "Model": name, "Seed": seed, "CBF_bin": b,
                "CBF_RMSE": np.sqrt(sub["CBF_err_sq"].mean()),
                "ATT_RMSE": np.sqrt(sub["ATT_err_sq"].mean()),
                "Count": len(sub)
            })

df_att = pd.DataFrame(results_att).groupby(["Model", "ATT_bin"]).agg(
    CBF_RMSE_mean=("CBF_RMSE", "mean"), CBF_RMSE_std=("CBF_RMSE", "std"),
    ATT_RMSE_mean=("ATT_RMSE", "mean"), ATT_RMSE_std=("ATT_RMSE", "std"),
).reset_index()

df_cbf = pd.DataFrame(results_cbf).groupby(["Model", "CBF_bin"]).agg(
    CBF_RMSE_mean=("CBF_RMSE", "mean"), CBF_RMSE_std=("CBF_RMSE", "std"),
    ATT_RMSE_mean=("ATT_RMSE", "mean"), ATT_RMSE_std=("ATT_RMSE", "std"),
).reset_index()

df_att.to_csv("architectures/02_hierarchical_kinetic_conditioning/tables/error_by_att_range.csv", index=False)
df_cbf.to_csv("architectures/02_hierarchical_kinetic_conditioning/tables/error_by_cbf_range.csv", index=False)

print("\nError by ATT Range:")
print(df_att.to_string(index=False))

print("\nError by CBF Range:")
print(df_cbf.to_string(index=False))
