import pandas as pd
import numpy as np

paths = {
    'Arch02': 'architectures/02_hierarchical_kinetic_conditioning/tables',
    'Arch03': 'architectures/03_gradient_isolated_kinetic_conditioning/tables',
    'Arch04': 'architectures/04_partial_gradient_kinetic_conditioning/tables'
}

agg_res = []
for name, p in paths.items():
    df = pd.read_csv(f'{p}/agg_results.csv')
    df['Model'] = name
    agg_res.append(df)
df_agg = pd.concat(agg_res)

print('--- OVERALL CBF RMSE ---')
df_inf = df_agg.groupby(['Model', 'SNR']).mean(numeric_only=True).reset_index()
for snr in [np.inf, 50, 20, 15, 10, 5]:
    sub = df_inf[df_inf['SNR'] == snr].set_index('Model')
    print(f'SNR {snr}: Arch02={sub.loc["Arch02", "CBF RMSE"]:.3f}, Arch03={sub.loc["Arch03", "CBF RMSE"]:.3f}, Arch04={sub.loc["Arch04", "CBF RMSE"]:.3f}')

regime_res = []
for name, p in paths.items():
    df = pd.read_csv(f'{p}/error_by_att_range.csv')
    df['Model'] = name
    regime_res.append(df)
df_regime = pd.concat(regime_res)

print('\n--- ATT REGIME (CBF RMSE) ---')
df_reg = df_regime.groupby(['Model', 'ATT_bin']).mean(numeric_only=True).reset_index()
for att_bin in ['0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0']:
    sub = df_reg[df_reg['ATT_bin'] == att_bin].set_index('Model')
    print(f'{att_bin}: Arch02={sub.loc["Arch02", "CBF_RMSE_mean"]:.3f}, Arch03={sub.loc["Arch03", "CBF_RMSE_mean"]:.3f}, Arch04={sub.loc["Arch04", "CBF_RMSE_mean"]:.3f}')

latent_res = []
for name, p in paths.items():
    if name in ['Arch02', 'Arch03']: 
        df = pd.read_csv('architectures/03_gradient_isolated_kinetic_conditioning/tables/latent_probe.csv')
        df = df[df['Architecture'] == name].copy()
    else:
        df = pd.read_csv(f'{p}/latent_probe.csv')
    df['Model'] = name
    latent_res.append(df)
df_lat = pd.concat(latent_res)

print('\n--- LATENT ANALYSIS ---')
df_l = df_lat.groupby('Model').mean(numeric_only=True)
print(df_l[['Test R2 CBF', 'Test R2 ATT']])
