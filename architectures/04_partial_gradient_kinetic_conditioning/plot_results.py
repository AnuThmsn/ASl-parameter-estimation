import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('architectures/04_partial_gradient_kinetic_conditioning/plots', exist_ok=True)

paths = {
    'Arch02': 'architectures/02_hierarchical_kinetic_conditioning/tables',
    'Arch03': 'architectures/03_gradient_isolated_kinetic_conditioning/tables',
    'Arch04': 'architectures/04_partial_gradient_kinetic_conditioning/tables'
}

agg_data = []
for name, p in paths.items():
    df = pd.read_csv(f"{p}/agg_results.csv")
    df['Model'] = name
    agg_data.append(df)
df_agg = pd.concat(agg_data)

df_inf = df_agg[df_agg['SNR'] < np.inf]
for model in paths.keys():
    sub = df_inf[df_inf['Model'] == model]
    plt.plot(sub['SNR'], sub['CBF RMSE'], marker='o', label=model)
plt.gca().invert_xaxis()
plt.title('CBF RMSE vs SNR')
plt.legend()
plt.savefig('architectures/04_partial_gradient_kinetic_conditioning/plots/cbf_rmse_vs_snr.png')
plt.close()

for model in paths.keys():
    sub = df_inf[df_inf['Model'] == model]
    plt.plot(sub['SNR'], sub['ATT RMSE'], marker='o', label=model)
plt.gca().invert_xaxis()
plt.title('ATT RMSE vs SNR')
plt.legend()
plt.savefig('architectures/04_partial_gradient_kinetic_conditioning/plots/att_rmse_vs_snr.png')
plt.close()

regime_data = []
for name, p in paths.items():
    df = pd.read_csv(f"{p}/error_by_att_range.csv")
    df['Model'] = name
    regime_data.append(df)
df_regime = pd.concat(regime_data)
