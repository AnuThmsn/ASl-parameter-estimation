import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('architectures/05_adaptive_gradient_kinetic_conditioning/plots', exist_ok=True)

# We will collect canonical results for 02, 03, 04, and add 05.
df_can_o = pd.read_csv('comparisons/canonical_evaluation/canonical_overall.csv')
df_can_a = pd.read_csv('comparisons/canonical_evaluation/canonical_att_regime.csv')

df_05_o = pd.read_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/prediction_results.csv')
df_05_o['Model'] = 'Arch05'
# average over seeds for Arch05
df_05_o_agg = df_05_o.groupby(['Model', 'SNR']).mean(numeric_only=True).reset_index()
df_can_o_agg = df_can_o.groupby(['Model', 'SNR']).mean(numeric_only=True).reset_index()
df_overall = pd.concat([df_can_o_agg, df_05_o_agg])

# Plot 1: CBF RMSE vs SNR
plt.figure()
df_inf = df_overall[df_overall['SNR'] < np.inf]
for m in ['Arch02', 'Arch03', 'Arch04', 'Arch05']:
    sub = df_inf[df_inf['Model'] == m]
    plt.plot(sub['SNR'], sub['CBF RMSE'], marker='o', label=m)
plt.gca().invert_xaxis()
plt.title('CBF RMSE vs SNR')
plt.legend()
plt.savefig('architectures/05_adaptive_gradient_kinetic_conditioning/plots/plot_1_cbf_rmse_vs_snr.png')
plt.close()

# Plot 2: ATT RMSE vs SNR
plt.figure()
for m in ['Arch02', 'Arch03', 'Arch04', 'Arch05']:
    sub = df_inf[df_inf['Model'] == m]
    plt.plot(sub['SNR'], sub['ATT RMSE'], marker='o', label=m)
plt.gca().invert_xaxis()
plt.title('ATT RMSE vs SNR')
plt.legend()
plt.savefig('architectures/05_adaptive_gradient_kinetic_conditioning/plots/plot_2_att_rmse_vs_snr.png')
plt.close()

# Plot 5, 6, 7, 8: Alpha behavior
df_alpha_att = pd.read_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_att_regime.csv')
df_alpha_att_inf = df_alpha_att[df_alpha_att['SNR'] == np.inf]
plt.figure()
plt.bar(df_alpha_att_inf['ATT_bin'], df_alpha_att_inf['mean_alpha'], yerr=df_alpha_att_inf['std_alpha'])
plt.title('Mean Alpha vs ATT Regime (SNR inf)')
plt.ylim(0, 1.0)
plt.savefig('architectures/05_adaptive_gradient_kinetic_conditioning/plots/plot_5_mean_alpha_vs_att.png')
plt.close()

df_alpha_snr = pd.read_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_snr.csv')
plt.figure()
# Replace inf with a dummy large number or string for plotting if necessary, but just plot it linearly without inf
sub_snr = df_alpha_snr[df_alpha_snr['SNR'] < np.inf].copy()
plt.plot(sub_snr['SNR'], sub_snr['mean_alpha'], marker='o')
plt.gca().invert_xaxis()
plt.title('Mean Alpha vs SNR')
plt.ylim(0, 1.0)
plt.savefig('architectures/05_adaptive_gradient_kinetic_conditioning/plots/plot_7_mean_alpha_vs_snr.png')
plt.close()

df_alpha_cbf = pd.read_csv('architectures/05_adaptive_gradient_kinetic_conditioning/tables/alpha_by_cbf_regime.csv')
df_alpha_cbf_inf = df_alpha_cbf[df_alpha_cbf['SNR'] == np.inf]
plt.figure()
plt.bar(df_alpha_cbf_inf['CBF_bin'], df_alpha_cbf_inf['mean_alpha'], yerr=df_alpha_cbf_inf['std_alpha'])
plt.title('Mean Alpha vs CBF Regime (SNR inf)')
plt.ylim(0, 1.0)
plt.savefig('architectures/05_adaptive_gradient_kinetic_conditioning/plots/plot_8_mean_alpha_vs_cbf.png')
plt.close()

print("Plots saved.")
