"""
evaluate.py - Evaluation script for Literature U-Net
Evaluates trained 4-PLD U-Net on held-out test volumes at multiple SNRs.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import generate_spatial_volume
from src.physics.noise_models import add_noise

import yaml

SNR_LEVELS = [np.inf, 50, 20, 15, 10, 5]
ATT_EDGES  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CBF_EDGES  = [0.0, 33.3, 66.6, 100.0]
CBF_LABELS = ["0-33", "33-66", "66-100"]


def att_bin(v):
    for i in range(len(ATT_EDGES)-1):
        if ATT_EDGES[i] <= v <= ATT_EDGES[i+1]:
            return f"{ATT_EDGES[i]:.1f}-{ATT_EDGES[i+1]:.1f}"
    return "Other"


def cbf_bin(v):
    for i, (lo, hi) in enumerate(zip(CBF_EDGES[:-1], CBF_EDGES[1:])):
        if lo <= v <= hi:
            return CBF_LABELS[i]
    return "Other"


def evaluate_model(model, norm, test_vols, snr, D, H, W):
    sig_mean = norm["sig_mean"]; sig_std = norm["sig_std"]
    cbf_mean = float(norm["cbf_mean"]); cbf_std = float(norm["cbf_std"])
    att_mean = float(norm["att_mean"]); att_std = float(norm["att_std"])

    all_cbf_err, all_att_err = [], []
    all_cbf_true, all_att_true = [], []
    all_cbf_pred, all_att_pred = [], []
    all_brain = []
    vol_rows = []

    model.eval()
    for vi, (sig, cbf_gt, att_gt, mask) in enumerate(test_vols):
        # Apply noise
        if np.isinf(snr):
            sig_noisy = sig.copy()
        else:
            noise_sd = 334.2039 / snr
            rng_n = np.random.default_rng(99 + vi)
            sig_noisy = add_noise(sig.transpose(1,2,3,0).reshape(-1,4), noise_sd, rng=rng_n).reshape(D,H,W,4).transpose(3,0,1,2)

        sig_norm = (sig_noisy - sig_mean) / sig_std
        x_t = torch.tensor(sig_norm[None], dtype=torch.float32)
        with torch.no_grad():
            out = model(x_t)  # (1, 2, D, H, W)
        cbf_pred = out[0,0].numpy() * cbf_std + cbf_mean
        att_pred = out[0,1].numpy() * att_std + att_mean

        brain = mask > 0.5
        cbf_err = cbf_pred[brain] - cbf_gt[brain]
        att_err = att_pred[brain] - att_gt[brain]

        cbf_rmse = float(np.sqrt(np.mean(cbf_err**2)))
        att_rmse = float(np.sqrt(np.mean(att_err**2)))
        cbf_mae  = float(np.mean(np.abs(cbf_err)))
        att_mae  = float(np.mean(np.abs(att_err)))
        cbf_bias = float(np.mean(cbf_err))
        att_bias = float(np.mean(att_err))

        vol_rows.append({"volume_id": vi, "snr": snr, "cbf_rmse": cbf_rmse, "att_rmse": att_rmse,
                          "cbf_mae": cbf_mae, "att_mae": att_mae, "cbf_bias": cbf_bias, "att_bias": att_bias})

        all_cbf_true.append(cbf_gt[brain])
        all_att_true.append(att_gt[brain])
        all_cbf_pred.append(cbf_pred[brain])
        all_att_pred.append(att_pred[brain])

    cbf_true_all = np.concatenate(all_cbf_true)
    att_true_all = np.concatenate(all_att_true)
    cbf_pred_all = np.concatenate(all_cbf_pred)
    att_pred_all = np.concatenate(all_att_pred)

    return vol_rows, cbf_true_all, att_true_all, cbf_pred_all, att_pred_all


def main():
    with open("experiments/literature_unet/configs/unet_config.yaml") as f:
        cfg = yaml.safe_load(f)

    D=cfg["data"]["D"]; H=cfg["data"]["H"]; W=cfg["data"]["W"]
    seeds = cfg["training"]["seeds"]

    # Generate fixed test volumes
    print("Generating test volumes...", flush=True)
    rng_te = np.random.default_rng(cfg["data"]["test_seed"])
    test_vols = []
    for _ in range(cfg["data"]["n_test"]):
        sig, cbf, att, mask = generate_spatial_volume(rng_te, D, H, W, noise_sd=0.0)
        test_vols.append((sig, cbf, att, mask))

    all_raw = []
    all_att_regime = []
    all_cbf_regime = []

    for seed in seeds:
        ckpt_path = f"experiments/literature_unet/checkpoints/unet_4pld_seed{seed}.pt"
        norm_path  = f"experiments/literature_unet/checkpoints/unet_norm_seed{seed}.npz"
        if not os.path.exists(ckpt_path):
            print(f"Missing checkpoint for seed {seed}, skipping."); continue

        norm = np.load(norm_path)
        model = LuciwUNet3D(in_channels=4, base_channels=cfg["model"]["base_channels"])
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        print(f"Evaluating seed {seed}...", flush=True)

        for snr in SNR_LEVELS:
            vol_rows, cbf_true, att_true, cbf_pred, att_pred = evaluate_model(
                model, norm, test_vols, snr, D, H, W)
            for vr in vol_rows:
                vr["seed"] = seed; all_raw.append(vr)

            # ATT regime
            for a_bin in [att_bin(v) for v in att_true]:
                pass  # will do below via vectorised
            for ab in [f"{ATT_EDGES[i]:.1f}-{ATT_EDGES[i+1]:.1f}" for i in range(len(ATT_EDGES)-1)]:
                mask2 = np.array([att_bin(v)==ab for v in att_true])
                if mask2.sum() > 0:
                    c_err = cbf_pred[mask2]-cbf_true[mask2]
                    a_err = att_pred[mask2]-att_true[mask2]
                    all_att_regime.append({"seed": seed, "snr": snr, "att_bin": ab,
                        "cbf_rmse": float(np.sqrt(np.mean(c_err**2))),
                        "att_rmse": float(np.sqrt(np.mean(a_err**2)))})

            # CBF regime
            for cb in CBF_LABELS:
                mask2 = np.array([cbf_bin(v)==cb for v in cbf_true])
                if mask2.sum() > 0:
                    c_err = cbf_pred[mask2]-cbf_true[mask2]
                    a_err = att_pred[mask2]-att_true[mask2]
                    all_cbf_regime.append({"seed": seed, "snr": snr, "cbf_bin": cb,
                        "cbf_rmse": float(np.sqrt(np.mean(c_err**2))),
                        "att_rmse": float(np.sqrt(np.mean(a_err**2)))})

    os.makedirs("experiments/literature_unet/results", exist_ok=True)
    df_raw = pd.DataFrame(all_raw)
    df_raw.to_csv("experiments/literature_unet/results/raw_results.csv", index=False)

    df_sum = df_raw.groupby("snr").agg(
        cbf_rmse_mean=("cbf_rmse","mean"), cbf_rmse_std=("cbf_rmse","std"),
        att_rmse_mean=("att_rmse","mean"), att_rmse_std=("att_rmse","std")).reset_index()
    df_sum["model"] = "unet_4pld"
    df_sum.to_csv("experiments/literature_unet/results/summary_results.csv", index=False)

    pd.DataFrame(all_att_regime).to_csv("experiments/literature_unet/results/att_regime_results.csv", index=False)
    pd.DataFrame(all_cbf_regime).to_csv("experiments/literature_unet/results/cbf_regime_results.csv", index=False)

    print("\n=== EVALUATION SUMMARY (mean over seeds & volumes) ===")
    print(df_sum[["snr","cbf_rmse_mean","att_rmse_mean"]].to_string(index=False))
    print("Results saved to experiments/literature_unet/results/")


if __name__ == "__main__":
    main()
