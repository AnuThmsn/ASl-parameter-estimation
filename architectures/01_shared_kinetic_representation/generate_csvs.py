"""
generate_csvs.py — Corrected evaluation pipeline for Architecture 01.

AUDIT FIX (2026-09-29):
  Previous version applied add_noise() to X_clean (already = 2S),
  which doubled the signal amplitude before noise, misrepresenting SNR.
  Correct pipeline: apply add_noise() to S_true (raw ASL signal).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.physics.noise_models import add_noise
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization
from src.training.evaluation import calc_metrics
from model import SharedKineticNet
from train import BaselineComboNet, AblationIndependentNet

SEEDS = [42, 123, 2024]
Y_MIN, Y_MAX = [0.0, 0.5], [100.0, 3.0]
SNRS = [np.inf, 50, 20, 15, 10, 5]
N_TRAIN = 20_000
N_TEST  = 5_000

results = []

for seed in SEEDS:
    print(f"\n=== Evaluating seed {seed} ===")
    rng_norm = np.random.default_rng(seed)

    # Reproduce training normalizers exactly: generate same N_TRAIN samples
    X_tr, Y_tr, _ = generate_data(N_TRAIN, rng_norm)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)

    # Generate a fixed held-out test ground truth (S_true, Y_ts)
    # Use seed=7 for test reproducibility, independent of training seed
    rng_test_gt = np.random.default_rng(7)
    cbf_test = rng_test_gt.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test_gt.uniform(ATT_MIN, ATT_MAX, N_TEST)
    S_true   = compute_signals_vec(cbf_test, att_test) * SCALE   # shape (N, 4)
    Y_ts     = np.stack([cbf_test, att_test], axis=1).astype(np.float32)

    # Load models
    models = {
        "Baseline":      BaselineComboNet(Y_mean, Y_std, Y_MIN, Y_MAX),
        "Architecture01": SharedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=Y_MIN, y_max=Y_MAX),
        "Ablation":      AblationIndependentNet(Y_mean, Y_std, Y_MIN, Y_MAX),
    }
    for name, model in models.items():
        ckpt = f"architectures/01_shared_kinetic_representation/checkpoints/{name}_seed{seed}.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()

    for snr in SNRS:
        # CORRECTED: apply add_noise to S_true, not to a pre-processed 2S
        rng_noise = np.random.default_rng(99)   # fixed noise seed for reproducibility
        if np.isinf(snr):
            # Clean: X = add_noise(S_true, 0) = 2 * S_true (consistent with training)
            noise_sd = 0.0
        else:
            noise_sd = 334.2039 / snr

        X_ts = add_noise(S_true, noise_sd, rng=rng_noise)   # shape (N, 4)
        X_ts_norm = apply_normalization(X_ts, X_mean, X_std)

        for name, model in models.items():
            phys_pred = model.predict_physical(torch.tensor(X_ts_norm).float()).numpy()
            cbf_m = calc_metrics(Y_ts[:, 0], phys_pred[:, 0])
            att_m = calc_metrics(Y_ts[:, 1], phys_pred[:, 1])
            results.append({
                "Seed": seed, "Model": name, "SNR": snr,
                "CBF RMSE": cbf_m["RMSE"], "CBF MAE": cbf_m["MAE"],
                "CBF R2":   cbf_m["R2"],   "CBF CCC": cbf_m["CCC"],
                "ATT RMSE": att_m["RMSE"], "ATT MAE": att_m["MAE"],
                "ATT R2":   att_m["R2"],   "ATT CCC": att_m["CCC"],
            })
            print(f"  {name:20s}  SNR={str(snr):>4s}  CBF RMSE={cbf_m['RMSE']:.3f}  ATT RMSE={att_m['RMSE']:.4f}")

# Save per-seed raw results
raw_df = pd.DataFrame(results)
raw_df.to_csv("architectures/01_shared_kinetic_representation/tables/raw_results.csv", index=False)

# Aggregate across seeds
agg_df = raw_df.groupby(["Model", "SNR"]).agg(
    CBF_RMSE_mean=("CBF RMSE", "mean"), CBF_RMSE_std=("CBF RMSE", "std"),
    CBF_MAE_mean=("CBF MAE",  "mean"), CBF_MAE_std=("CBF MAE",  "std"),
    CBF_R2_mean=("CBF R2",    "mean"),
    CBF_CCC_mean=("CBF CCC",  "mean"),
    ATT_RMSE_mean=("ATT RMSE","mean"), ATT_RMSE_std=("ATT RMSE","std"),
    ATT_MAE_mean=("ATT MAE",  "mean"), ATT_MAE_std=("ATT MAE",  "std"),
    ATT_R2_mean=("ATT R2",    "mean"),
    ATT_CCC_mean=("ATT CCC",  "mean"),
).reset_index()

agg_df.to_csv("comparisons/architecture_00_vs_01.csv", index=False)
print("\nSaved raw_results.csv and architecture_00_vs_01.csv")
print("\nAggregated results (mean across 3 seeds):")
print(agg_df[["Model","SNR","CBF_RMSE_mean","ATT_RMSE_mean"]].to_string(index=False))
