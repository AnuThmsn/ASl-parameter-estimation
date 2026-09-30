import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('final_training_audit/figures/convergence', exist_ok=True)
os.makedirs('final_training_audit/audit', exist_ok=True)

# 1. Convergence Summary and Previous vs Final
conv = pd.read_csv('final_training_audit/audit/convergence_summary.csv')
mean_best_epochs = conv.groupby('Model')['Best epoch'].mean().round(1)

# Generate Convergence Plots
models = conv['Model'].unique()
for m in models:
    plt.figure(figsize=(10, 6))
    for seed in [42, 123, 2024]:
        try:
            hist_dir = "final_training_audit/histories/fixed_alpha" if "fixed_alpha" in m else f"final_training_audit/histories/{m.split('_')[0]}"
            df_hist = pd.read_csv(f"{hist_dir}/{m}_seed{seed}.csv")
            plt.plot(df_hist['epoch'], df_hist['train_loss'], label=f'Train S{seed}', linestyle='--', alpha=0.6)
            plt.plot(df_hist['epoch'], df_hist['val_loss'], label=f'Val S{seed}', alpha=0.8)
        except Exception as e:
            pass
    plt.title(f"{m} Convergence")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'final_training_audit/figures/convergence/{m}_convergence.png')
    plt.close()

# 2. Final Results Table
mean_std = pd.read_csv('final_training_audit/results/mean_std_results.csv')

# Format means and stds for Markdown tables
def fmt(m, s): return f"{m:.3f} ± {s:.3f}"

archs = ["arch02", "arch03", "arch04", "arch05"]
final_table = []
for arch in archs:
    row = {"Architecture": arch, "Parameters": "~11k", "Training samples": 20000}
    # Best epoch
    row["Best epoch"] = mean_best_epochs.get(arch, "-")
    # Clean (inf)
    d_inf = mean_std[(mean_std['Model'] == arch) & (mean_std['SNR'] == float('inf'))]
    if not d_inf.empty:
        row["Clean CBF RMSE"] = fmt(d_inf['CBF_RMSE_mean'].values[0], d_inf['CBF_RMSE_std'].values[0])
        row["Clean ATT RMSE"] = fmt(d_inf['ATT_RMSE_mean'].values[0], d_inf['ATT_RMSE_std'].values[0])
        
    for snr in [50, 10, 5]:
        d_snr = mean_std[(mean_std['Model'] == arch) & (mean_std['SNR'] == float(snr))]
        if not d_snr.empty:
            row[f"SNR{snr} CBF RMSE"] = fmt(d_snr['CBF_RMSE_mean'].values[0], d_snr['CBF_RMSE_std'].values[0])
            row[f"SNR{snr} ATT RMSE"] = fmt(d_snr['ATT_RMSE_mean'].values[0], d_snr['ATT_RMSE_std'].values[0])
    final_table.append(row)

df_final_table = pd.DataFrame(final_table)
df_final_table.to_csv('final_training_audit/results/architecture_comparison.csv', index=False)

# 3. Fixed Alpha Table
fixed_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_table = []
for a in fixed_alphas:
    m_name = f"fixed_alpha_a{a:.2f}"
    d_inf = mean_std[(mean_std['Model'] == m_name) & (mean_std['SNR'] == float('inf'))]
    if not d_inf.empty:
        alpha_table.append({
            "alpha": a,
            "CBF RMSE mean ± SD": fmt(d_inf['CBF_RMSE_mean'].values[0], d_inf['CBF_RMSE_std'].values[0]),
            "ATT RMSE mean ± SD": fmt(d_inf['ATT_RMSE_mean'].values[0], d_inf['ATT_RMSE_std'].values[0])
        })
# And add Arch05 (Adaptive)
d_inf_05 = mean_std[(mean_std['Model'] == "arch05") & (mean_std['SNR'] == float('inf'))]
if not d_inf_05.empty:
    alpha_table.append({
        "alpha": "adaptive",
        "CBF RMSE mean ± SD": fmt(d_inf_05['CBF_RMSE_mean'].values[0], d_inf_05['CBF_RMSE_std'].values[0]),
        "ATT RMSE mean ± SD": fmt(d_inf_05['ATT_RMSE_mean'].values[0], d_inf_05['ATT_RMSE_std'].values[0])
    })

df_alpha = pd.DataFrame(alpha_table)
df_alpha.to_csv('final_training_audit/results/fixed_alpha_results.csv', index=False)

# Generate Markdown file content parts
print("Markdown components generated successfully.")
