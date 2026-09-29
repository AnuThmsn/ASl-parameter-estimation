import pandas as pd
import numpy as np
import os

df_o = pd.read_csv('comparisons/canonical_evaluation/canonical_overall.csv')
df_a = pd.read_csv('comparisons/canonical_evaluation/canonical_att_regime.csv')
df_l = pd.read_csv('comparisons/canonical_evaluation/canonical_latent.csv')

print('--- OVERALL CBF RMSE ---')
df_o_inf = df_o.groupby(['Model', 'SNR']).mean(numeric_only=True).reset_index()
for snr in [np.inf, 50, 20, 15, 10, 5]:
    sub = df_o_inf[df_o_inf['SNR'] == snr].set_index('Model')
    print(f'SNR {snr}: Arch02={sub.loc["Arch02", "CBF RMSE"]:.3f}, Arch03={sub.loc["Arch03", "CBF RMSE"]:.3f}, Arch04={sub.loc["Arch04", "CBF RMSE"]:.3f}')

print('\n--- OVERALL ATT RMSE ---')
for snr in [np.inf, 50, 20, 15, 10, 5]:
    sub = df_o_inf[df_o_inf['SNR'] == snr].set_index('Model')
    print(f'SNR {snr}: Arch02={sub.loc["Arch02", "ATT RMSE"]:.3f}, Arch03={sub.loc["Arch03", "ATT RMSE"]:.3f}, Arch04={sub.loc["Arch04", "ATT RMSE"]:.3f}')

print('\n--- ATT REGIME CBF RMSE (SNR inf) ---')
df_a_inf = df_a[df_a['SNR'] == np.inf].groupby(['Model', 'ATT_bin']).mean(numeric_only=True).reset_index()
for att_bin in ['0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0']:
    sub = df_a_inf[df_a_inf['ATT_bin'] == att_bin].set_index('Model')
    print(f'{att_bin}: Arch02={sub.loc["Arch02", "CBF RMSE"]:.3f}, Arch03={sub.loc["Arch03", "CBF RMSE"]:.3f}, Arch04={sub.loc["Arch04", "CBF RMSE"]:.3f}')

print('\n--- LATENT ANALYSIS (Test R2) ---')
df_l_mean = df_l.groupby('Model').mean(numeric_only=True)
print(df_l_mean[['ATT R2', 'CBF R2']])
