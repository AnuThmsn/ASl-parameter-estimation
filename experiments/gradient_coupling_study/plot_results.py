import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

plt.style.use('ggplot')
os.makedirs('experiments/gradient_coupling_study/figures', exist_ok=True)

df_sum = pd.read_csv('experiments/gradient_coupling_study/results/summary_results.csv')
df_sum.replace(np.inf, 9999, inplace=True) # for plotting

# Color mapping
models = ['alpha_0.00', 'alpha_0.25', 'alpha_0.50', 'alpha_0.75', 'alpha_1.00', 'Adaptive']
cmap = plt.get_cmap('viridis')
colors = {f'alpha_{a:.2f}': cmap(a) for a in [0.00, 0.25, 0.50, 0.75, 1.00]}
colors['Adaptive'] = 'red'

# Figure 1: CBF RMSE vs SNR
plt.figure(figsize=(8, 6))
for m in models:
    sub = df_sum[(df_sum['model'] == m) & (df_sum['snr'] != 9999)]
    plt.errorbar(sub['snr'], sub['cbf_rmse_mean'], yerr=sub['cbf_rmse_std'], label=m, color=colors[m], marker='o')
plt.gca().invert_xaxis()
plt.title('CBF RMSE vs SNR')
plt.xlabel('SNR')
plt.ylabel('CBF RMSE')
plt.legend()
plt.savefig('experiments/gradient_coupling_study/figures/cbf_rmse_vs_snr.png')
plt.close()

# Figure 2: ATT RMSE vs SNR
plt.figure(figsize=(8, 6))
for m in models:
    sub = df_sum[(df_sum['model'] == m) & (df_sum['snr'] != 9999)]
    plt.errorbar(sub['snr'], sub['att_rmse_mean'], yerr=sub['att_rmse_std'], label=m, color=colors[m], marker='o')
plt.gca().invert_xaxis()
plt.title('ATT RMSE vs SNR')
plt.xlabel('SNR')
plt.ylabel('ATT RMSE')
plt.legend()
plt.savefig('experiments/gradient_coupling_study/figures/att_rmse_vs_snr.png')
plt.close()

# Figure 3: Fixed alpha sweep
plt.figure(figsize=(8, 6))
fixed_alphas = [0.00, 0.25, 0.50, 0.75, 1.00]
snrs_to_plot = [9999, 20, 10, 5]
for snr in snrs_to_plot:
    y_means, y_stds = [], []
    for a in fixed_alphas:
        sub = df_sum[(df_sum['model'] == f'alpha_{a:.2f}') & (df_sum['snr'] == snr)]
        y_means.append(sub['cbf_rmse_mean'].values[0])
        y_stds.append(sub['cbf_rmse_std'].values[0])
    p = plt.errorbar(fixed_alphas, y_means, yerr=y_stds, marker='s', label=f'SNR={snr if snr!=9999 else "inf"}')
    
    # Overlay Adaptive
    sub_ad = df_sum[(df_sum['model'] == 'Adaptive') & (df_sum['snr'] == snr)]
    ad_mean = sub_ad['cbf_rmse_mean'].values[0]
    plt.axhline(ad_mean, color=p[0].get_color(), linestyle='--')

plt.title('Fixed Alpha Sweep vs Adaptive (dashed)')
plt.xlabel('Fixed Alpha')
plt.ylabel('CBF RMSE')
plt.legend()
plt.savefig('experiments/gradient_coupling_study/figures/fixed_alpha_sweep.png')
plt.close()

# Load alpha samples
df_alpha = pd.read_csv('experiments/gradient_coupling_study/results/alpha_samples.csv')
df_alpha_inf = df_alpha[df_alpha['snr'] == 9999] if 'snr' in df_alpha.columns else df_alpha.head(15000)
# actually alpha_samples has columns: Seed, SNR, Sample_ID, CBF, ATT, Alpha
df_alpha_inf = df_alpha[df_alpha['SNR'] == np.inf]

# Figure 4: Alpha distribution
plt.figure(figsize=(8, 6))
plt.hist(df_alpha_inf['Alpha'], bins=50, color='red', alpha=0.7, edgecolor='black')
plt.title('Distribution of Adaptive Alpha (SNR=inf)')
plt.xlabel('Alpha')
plt.ylabel('Frequency')
plt.savefig('experiments/gradient_coupling_study/figures/alpha_distribution.png')
plt.close()

# Figure 5: Alpha vs ATT
plt.figure(figsize=(8, 6))
plt.scatter(df_alpha_inf['ATT'], df_alpha_inf['Alpha'], alpha=0.1, s=2)
# Add rolling mean
att_sorted = df_alpha_inf.sort_values('ATT')
rolling_mean = att_sorted['Alpha'].rolling(500, center=True).mean()
plt.plot(att_sorted['ATT'], rolling_mean, color='black', linewidth=2, label='Rolling Mean')
plt.title('Adaptive Alpha vs ATT')
plt.xlabel('ATT (s)')
plt.ylabel('Alpha')
plt.legend()
plt.savefig('experiments/gradient_coupling_study/figures/alpha_vs_att.png')
plt.close()

# Merge identifiability
df_ident = pd.read_csv('experiments/gradient_coupling_study/results/identifiability_results.csv')
# df_alpha_inf has 3 seeds, so average alpha across seeds per sample
df_alpha_avg = df_alpha_inf.groupby('Sample_ID')['Alpha'].mean().reset_index()
df_merged = pd.merge(df_alpha_avg, df_ident, on='Sample_ID')

# Figure 6: Alpha vs identifiability (Log Condition Number)
plt.figure(figsize=(8, 6))
plt.scatter(df_merged['Log_Condition_Number'], df_merged['Alpha'], alpha=0.3, s=5)
r, p = pearsonr(df_merged['Log_Condition_Number'], df_merged['Alpha'])
plt.title(f'Alpha vs Log Condition Number (r={r:.3f})')
plt.xlabel('Log(Condition Number)')
plt.ylabel('Alpha')
plt.savefig('experiments/gradient_coupling_study/figures/alpha_vs_identifiability.png')
plt.close()

# Save Identifiability correlations to CSV
corr_results = []
metrics = ['Log_Condition_Number', 'Sigma_Min', 'Sensitivity_Angle', 'ATT', 'CBF']
for m in metrics:
    rp, _ = pearsonr(df_merged[m], df_merged['Alpha'])
    rs, _ = spearmanr(df_merged[m], df_merged['Alpha'])
    corr_results.append({'Metric': m, 'Pearson_r': rp, 'Spearman_r': rs})

df_corr = pd.DataFrame(corr_results)
df_corr.to_csv('experiments/gradient_coupling_study/results/alpha_identifiability_correlations.csv', index=False)

# Figure 7: ATT Regime comparison
df_att = pd.read_csv('experiments/gradient_coupling_study/results/att_regime_results.csv')
df_att_sum = df_att.groupby(['model', 'snr', 'ATT_bin']).agg(cbf_rmse_mean=('cbf_rmse', 'mean')).reset_index()
df_att_inf = df_att_sum[df_att_sum['snr'] == np.inf]

plt.figure(figsize=(10, 6))
width = 0.15
x = np.arange(len(df_att_inf['ATT_bin'].unique()))
for i, m in enumerate(models):
    sub = df_att_inf[df_att_inf['model'] == m]
    plt.bar(x + i*width, sub['cbf_rmse_mean'], width, label=m, color=colors[m])
plt.xticks(x + width*2.5, df_att_inf['ATT_bin'].unique())
plt.title('CBF RMSE by ATT Regime (SNR=inf)')
plt.xlabel('ATT Regime (s)')
plt.ylabel('CBF RMSE')
plt.legend()
plt.savefig('experiments/gradient_coupling_study/figures/att_regime_comparison.png')
plt.close()

# Distribution stats
dist_stats = {
    'mean': df_alpha_inf['Alpha'].mean(),
    'median': df_alpha_inf['Alpha'].median(),
    'std': df_alpha_inf['Alpha'].std(),
    'min': df_alpha_inf['Alpha'].min(),
    'max': df_alpha_inf['Alpha'].max(),
    '10th': df_alpha_inf['Alpha'].quantile(0.10),
    '25th': df_alpha_inf['Alpha'].quantile(0.25),
    '75th': df_alpha_inf['Alpha'].quantile(0.75),
    '90th': df_alpha_inf['Alpha'].quantile(0.90)
}
pd.DataFrame([dist_stats]).to_csv('experiments/gradient_coupling_study/results/alpha_distribution_stats.csv', index=False)

print("Plots and analysis complete.")
