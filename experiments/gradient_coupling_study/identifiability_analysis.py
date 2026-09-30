import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX

def compute_jacobian(cbf, att):
    eps_cbf = 1e-3
    eps_att = 1e-4
    S0 = compute_signals_vec(cbf, att)
    S_cbf = compute_signals_vec(cbf + eps_cbf, att)
    S_att = compute_signals_vec(cbf, att + eps_att)
    
    dS_dcbf = (S_cbf - S0) / eps_cbf  # (N, 4)
    dS_datt = (S_att - S0) / eps_att  # (N, 4)
    
    J = np.stack([dS_dcbf, dS_datt], axis=2)
    return J, dS_dcbf, dS_datt

def get_bin(val, edges):
    for i in range(len(edges)-1):
        if edges[i] <= val <= edges[i+1]:
            return f"{edges[i]:.1f}-{edges[i+1]:.1f}"
    return "Out"

def main():
    rng_test = np.random.default_rng(7)
    N_TEST = 5_000
    cbf_test = rng_test.uniform(CBF_MIN, CBF_MAX, N_TEST)
    att_test = rng_test.uniform(ATT_MIN, ATT_MAX, N_TEST)
    
    J, dS_dcbf, dS_datt = compute_jacobian(cbf_test, att_test)
    
    cond_nums = []
    sigma_mins = []
    sigma_maxs = []
    angles = []
    
    for i in range(N_TEST):
        Ji = J[i]
        U, S, Vh = np.linalg.svd(Ji, full_matrices=False)
        s_max, s_min = S[0], S[1]
        
        sigma_maxs.append(s_max)
        sigma_mins.append(s_min)
        cond_nums.append(s_max / s_min if s_min > 1e-12 else np.inf)
        
        u = dS_dcbf[i]
        v = dS_datt[i]
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        
        if nu > 1e-12 and nv > 1e-12:
            cos_theta = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
            angles.append(np.arccos(cos_theta))
        else:
            angles.append(np.pi / 2.0)
            
    df = pd.DataFrame({
        'Sample_ID': np.arange(N_TEST),
        'CBF': cbf_test,
        'ATT': att_test,
        'Condition_Number': cond_nums,
        'Log_Condition_Number': np.log10(cond_nums),
        'Sigma_Min': sigma_mins,
        'Sigma_Max': sigma_maxs,
        'Sensitivity_Angle': angles
    })
    
    att_edges = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    df['ATT_bin'] = df['ATT'].apply(lambda x: get_bin(x, att_edges))
    
    os.makedirs('experiments/gradient_coupling_study/results', exist_ok=True)
    df.to_csv('experiments/gradient_coupling_study/results/identifiability_results.csv', index=False)
    print("Identifiability analysis complete.")

if __name__ == '__main__':
    main()
