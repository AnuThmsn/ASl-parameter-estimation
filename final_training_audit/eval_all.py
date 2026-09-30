import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import apply_normalization, apply_target_standardization
from src.physics.noise_models import add_noise
from final_training_audit.train_all import build_model

def get_att_bin(att, bins):
    for i in range(len(bins)-1):
        if bins[i] <= att <= bins[i+1]:
            return f"{bins[i]:.1f}-{bins[i+1]:.1f}"
    return "Other"

def get_cbf_bin(cbf, bins):
    labels = ["0-33", "33-66", "66-100"]
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= cbf <= hi:
            return labels[i]
    return "Other"

def main():
    with open("final_training_audit/configs/final_training_config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Starting Final Evaluation Audit ({device}) ===")
    
    norm_file = "final_training_audit/configs/canonical_normalization.npz"
    if not os.path.exists(norm_file):
        raise FileNotFoundError("Run training first to generate normalization stats.")
    
    norms = np.load(norm_file)
    x_mean, x_std = norms["x_mean"], norms["x_std"]
    y_mean, y_std = norms["y_mean"], norms["y_std"]
    
    rng_te = np.random.default_rng(cfg['data']['test_seed'])
    X_test_clean, y_test, _ = generate_data(cfg['data']['n_test'], rng_te, 0, 0.0)
    
    tasks = []
    for arch in ["arch02", "arch03", "arch04", "arch05"]:
        for seed in cfg['training']['seeds']:
            tasks.append((arch, seed, None))
            
    for alpha in cfg['fixed_alphas']:
        for seed in cfg['training']['seeds']:
            tasks.append(("fixed_alpha", seed, alpha))
            
    all_raw_results = []
    all_att_regime = []
    all_cbf_regime = []
    alpha_records = []
    
    for arch, seed, alpha in tasks:
        task_name = f"{arch}_seed{seed}" if alpha is None else f"{arch}_a{alpha:.2f}_seed{seed}"
        model_name = f"{arch}_a{alpha:.2f}" if alpha is not None else arch
        ckpt_dir = f"final_training_audit/checkpoints/{arch}" if alpha is None else "final_training_audit/checkpoints/fixed_alpha"
        ckpt_path = f"{ckpt_dir}/{task_name}.pt"
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} missing. Skipping.")
            continue
            
        print(f"Evaluating {task_name}...")
        model = build_model(arch, alpha)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        for snr in cfg['evaluation']['snr_levels']:
            if np.isinf(float(snr)):
                X_noisy = X_test_clean.copy()
            else:
                sd = 334.2039 / float(snr)
                rng_n = np.random.default_rng(44 + int(snr))
                X_noisy = add_noise(X_test_clean, np.full(X_test_clean.shape[0], sd)[:, None], rng=rng_n)
                
            X_norm = apply_normalization(X_noisy, x_mean, x_std)
            
            with torch.no_grad():
                inp = torch.tensor(X_norm, dtype=torch.float32).to(device)
                out = model(inp)
                pred_norm = out[0]
                
                # If Arch05, record alpha
                if arch == "arch05" and hasattr(model, 'gate'):
                    alpha_vals = torch.sigmoid(model.gate(model.shared_encoder(inp))).cpu().numpy()
                    for v in alpha_vals:
                        alpha_records.append({'seed': seed, 'snr': snr, 'alpha': float(v)})
                
                pred = pred_norm.cpu().numpy() * y_std + y_mean
                
            err = pred - y_test
            cbf_err = err[:, 0]
            att_err = err[:, 1]
            
            cbf_rmse = np.sqrt(np.mean(cbf_err**2))
            att_rmse = np.sqrt(np.mean(att_err**2))
            cbf_mae = np.mean(np.abs(cbf_err))
            att_mae = np.mean(np.abs(att_err))
            cbf_bias = np.mean(cbf_err)
            att_bias = np.mean(att_err)
            
            all_raw_results.append({
                "Model": model_name,
                "Seed": seed,
                "SNR": float(snr),
                "CBF RMSE": float(cbf_rmse),
                "ATT RMSE": float(att_rmse),
                "CBF MAE": float(cbf_mae),
                "ATT MAE": float(att_mae),
                "CBF Bias": float(cbf_bias),
                "ATT Bias": float(att_bias)
            })
            
            # ATT regime analysis
            att_gt = y_test[:, 1]
            for i in range(len(cfg['evaluation']['att_bins'])-1):
                lo = float(cfg['evaluation']['att_bins'][i])
                hi = float(cfg['evaluation']['att_bins'][i+1])
                a_bin = f"{lo:.1f}-{hi:.1f}"
                mask = (att_gt >= lo) & (att_gt <= hi)
                if mask.sum() > 0:
                    a_err_r = att_err[mask]
                    c_err_r = cbf_err[mask]
                    all_att_regime.append({
                        "Model": model_name, "Seed": seed, "SNR": float(snr), "ATT Regime": a_bin,
                        "CBF RMSE": float(np.sqrt(np.mean(c_err_r**2))),
                        "ATT RMSE": float(np.sqrt(np.mean(a_err_r**2)))
                    })
                    
            # CBF regime analysis
            cbf_gt = y_test[:, 0]
            cbf_labels = ["0-33", "33-66", "66-100"]
            cbf_edges = cfg['evaluation']['cbf_bins']
            for i, c_bin in enumerate(cbf_labels):
                lo = float(cbf_edges[i])
                hi = float(cbf_edges[i+1])
                mask = (cbf_gt >= lo) & (cbf_gt <= hi)
                if mask.sum() > 0:
                    a_err_r = att_err[mask]
                    c_err_r = cbf_err[mask]
                    all_cbf_regime.append({
                        "Model": model_name, "Seed": seed, "SNR": float(snr), "CBF Regime": c_bin,
                        "CBF RMSE": float(np.sqrt(np.mean(c_err_r**2))),
                        "ATT RMSE": float(np.sqrt(np.mean(a_err_r**2)))
                    })
                    
    # Save overall raw results
    df_raw = pd.DataFrame(all_raw_results)
    os.makedirs("final_training_audit/results", exist_ok=True)
    df_raw.to_csv("final_training_audit/results/overall_results.csv", index=False)
    
    # Save mean/std results
    df_mean_std = df_raw.groupby(["Model", "SNR"]).agg(
        CBF_RMSE_mean=("CBF RMSE", "mean"),
        CBF_RMSE_std=("CBF RMSE", "std"),
        ATT_RMSE_mean=("ATT RMSE", "mean"),
        ATT_RMSE_std=("ATT RMSE", "std"),
        CBF_MAE_mean=("CBF MAE", "mean"),
        CBF_MAE_std=("CBF MAE", "std")
    ).reset_index()
    
    df_mean_std.to_csv("final_training_audit/results/mean_std_results.csv", index=False)
    
    # Save regimes
    pd.DataFrame(all_att_regime).groupby(["Model", "SNR", "ATT Regime"]).mean().reset_index().to_csv("final_training_audit/results/att_regime_results.csv", index=False)
    pd.DataFrame(all_cbf_regime).groupby(["Model", "SNR", "CBF Regime"]).mean().reset_index().to_csv("final_training_audit/results/cbf_regime_results.csv", index=False)
    
    if alpha_records:
        pd.DataFrame(alpha_records).to_csv("final_training_audit/results/alpha_samples.csv", index=False)
    
    # Compute Paired Differences
    # Target baseline: arch05
    paired_diffs = []
    df_arch05 = df_raw[df_raw['Model'] == 'arch05'].set_index(['Seed', 'SNR'])
    
    for model_name in df_raw['Model'].unique():
        if model_name == 'arch05': continue
        df_m = df_raw[df_raw['Model'] == model_name].set_index(['Seed', 'SNR'])
        
        for (seed, snr), row_m in df_m.iterrows():
            if (seed, snr) in df_arch05.index:
                row_05 = df_arch05.loc[(seed, snr)]
                diff_cbf = row_m['CBF RMSE'] - row_05['CBF RMSE']
                diff_att = row_m['ATT RMSE'] - row_05['ATT RMSE']
                paired_diffs.append({
                    "Comparison": f"{model_name} - arch05",
                    "Seed": seed,
                    "SNR": snr,
                    "Delta CBF RMSE": diff_cbf,
                    "Delta ATT RMSE": diff_att
                })
                
    df_paired = pd.DataFrame(paired_diffs)
    if not df_paired.empty:
        df_paired.to_csv("final_training_audit/results/paired_differences.csv", index=False)
        
        # Summarize diffs
        df_paired_summary = df_paired.groupby(["Comparison", "SNR"]).agg(
            mean_delta_cbf=("Delta CBF RMSE", "mean"),
            std_delta_cbf=("Delta CBF RMSE", "std"),
            mean_delta_att=("Delta ATT RMSE", "mean"),
            std_delta_att=("Delta ATT RMSE", "std")
        ).reset_index()
        df_paired_summary['95CI_CBF'] = 1.96 * df_paired_summary['std_delta_cbf'] / np.sqrt(3)
        df_paired_summary.to_csv("final_training_audit/results/paired_differences_summary.csv", index=False)
        
    print("Evaluation Complete. Results saved to final_training_audit/results/")

if __name__ == "__main__":
    main()
