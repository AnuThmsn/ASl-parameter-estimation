import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, PLDs, tau, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, lmbda, alpha, beta, T1t, T1a
from src.physics.noise_models import calculate_noise_sd_from_snr, add_noise

OUT_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
REV_DIR = os.path.join(OUT_DIR, "review_package")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(REV_DIR, exist_ok=True)

mean_CBF, std_CBF = 50.0, 100.0 / np.sqrt(12)
mean_ATT, std_ATT = 1.75, 2.5 / np.sqrt(12)

def analytical_jacobian(cbf, att):
    cbf_expanded = cbf[:, None]
    att_expanded = att[:, None]
    plds = PLDs[None, :]
    
    S = compute_signals_vec(cbf, att) * SCALE
    dS_dCBF = np.zeros_like(S)
    mask = cbf_expanded[:, 0] > 1e-12
    dS_dCBF[mask] = S[mask] / cbf_expanded[mask]
    
    f_per_s = cbf_expanded / (6000.0 * lmbda)
    prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s * SCALE
    e_att = np.exp(-att_expanded / T1a)
    K_prime = prefix * e_att
    
    plds_exp = np.broadcast_to(plds, S.shape)
    
    term1_deriv = np.zeros_like(S)
    m1 = att_expanded < plds_exp
    term1_deriv[m1] = (1.0 / T1t) * np.exp(-(plds_exp[m1] - att_expanded.repeat(4, axis=1)[m1]) / T1t)
    
    term2_deriv = np.zeros_like(S)
    m2 = att_expanded < (plds_exp + tau)
    term2_deriv[m2] = (1.0 / T1t) * np.exp(-(plds_exp[m2] + tau - att_expanded.repeat(4, axis=1)[m2]) / T1t)
    
    dS_dATT = (-1.0 / T1a) * S + K_prime * (term1_deriv - term2_deriv)
    
    return dS_dCBF, dS_dATT

print("Running Experiment A & B (Finite-Difference Stability & Analytical Jacobian)...")
cbf_pts = [20.0, 50.0, 80.0]
att_pts = [0.5, 1.0, 1.5, 1.6, 2.0, 2.1, 2.5, 2.6, 3.0]
cbf_grid, att_grid = np.meshgrid(cbf_pts, att_pts)
cbf_flat, att_flat = cbf_grid.flatten(), att_grid.flatten()

h_cbf_list = [0.1, 0.5, 1.0, 2.0]
h_att_list = [0.001, 0.005, 0.01, 0.02, 0.05]

S_base = compute_signals_vec(cbf_flat, att_flat) * SCALE
J_ana_cbf, J_ana_att = analytical_jacobian(cbf_flat, att_flat)

fd_records = []
for h_cbf in h_cbf_list:
    for h_att in h_att_list:
        S_cbf_plus = compute_signals_vec(cbf_flat + h_cbf, att_flat) * SCALE
        S_att_plus = compute_signals_vec(cbf_flat, att_flat + h_att) * SCALE
        J_num_cbf = (S_cbf_plus - S_base) / h_cbf
        J_num_att = (S_att_plus - S_base) / h_att
        
        for i in range(len(cbf_flat)):
            err_cbf = np.linalg.norm(J_num_cbf[i] - J_ana_cbf[i]) / (np.linalg.norm(J_ana_cbf[i]) + 1e-12)
            err_att = np.linalg.norm(J_num_att[i] - J_ana_att[i]) / (np.linalg.norm(J_ana_att[i]) + 1e-12)
            
            J_num = np.column_stack([J_num_cbf[i] * std_CBF, J_num_att[i] * std_ATT])
            U, S_svd, Vh = np.linalg.svd(J_num)
            cond = S_svd[0] / S_svd[1] if S_svd[1] > 1e-12 else np.inf
            n1, n2 = np.linalg.norm(J_num[:,0]), np.linalg.norm(J_num[:,1])
            corr = np.dot(J_num[:,0], J_num[:,1]) / (n1 * n2) if n1>0 and n2>0 else 0
            
            fd_records.append({
                'CBF': cbf_flat[i], 'ATT': att_flat[i],
                'h_CBF': h_cbf, 'h_ATT': h_att,
                'err_CBF': err_cbf, 'err_ATT': err_att,
                'cond_num': cond, 'correlation': corr
            })
pd.DataFrame(fd_records).to_csv(os.path.join(TAB_DIR, "finite_difference_stability.csv"), index=False)

print("Running Experiment C (Signal Manifold Geometry)...")
cbf_dense = np.linspace(10, 100, 50)
att_dense = np.linspace(0.5, 3.0, 50)
C_dense, A_dense = np.meshgrid(cbf_dense, att_dense)
C_dense_flat, A_dense_flat = C_dense.flatten(), A_dense.flatten()
S_dense = compute_signals_vec(C_dense_flat, A_dense_flat) * SCALE

pd.DataFrame({
    'CBF': C_dense_flat, 'ATT': A_dense_flat,
    'S1': S_dense[:,0], 'S2': S_dense[:,1], 'S3': S_dense[:,2], 'S4': S_dense[:,3]
}).to_csv(os.path.join(TAB_DIR, "parameter_signal_grid.csv"), index=False)

# PCA Visualization
pca = PCA(n_components=2)
S_pca = pca.fit_transform(S_dense)
plt.figure(figsize=(8,6)); sc = plt.scatter(S_pca[:,0], S_pca[:,1], c=C_dense_flat, cmap='viridis', s=10)
plt.colorbar(sc, label='CBF'); plt.title('Signal Manifold (PCA) - CBF'); plt.savefig(os.path.join(FIG_DIR, "signal_manifold_pca_cbf.png")); plt.close()

plt.figure(figsize=(8,6)); sc = plt.scatter(S_pca[:,0], S_pca[:,1], c=A_dense_flat, cmap='plasma', s=10)
plt.colorbar(sc, label='ATT'); plt.title('Signal Manifold (PCA) - ATT'); plt.savefig(os.path.join(FIG_DIR, "signal_manifold_pca_att.png")); plt.close()

print("Running Experiment D (Local CBF/ATT Geometry)...")
cos_theta = np.zeros(len(C_dense_flat))
J_ana_cbf_dense, J_ana_att_dense = analytical_jacobian(C_dense_flat, A_dense_flat)
for i in range(len(C_dense_flat)):
    n1, n2 = np.linalg.norm(J_ana_cbf_dense[i]), np.linalg.norm(J_ana_att_dense[i])
    if n1 > 0 and n2 > 0:
        cos_theta[i] = np.dot(J_ana_cbf_dense[i], J_ana_att_dense[i]) / (n1 * n2)
plt.figure(figsize=(8,6)); plt.imshow(np.abs(cos_theta.reshape(C_dense.shape)), origin='lower', extent=[cbf_dense[0], cbf_dense[-1], att_dense[0], att_dense[-1]], aspect='auto', cmap='coolwarm'); plt.colorbar(label='|cos(theta)|')
plt.title('Cosine Similarity between CBF and ATT directions'); plt.xlabel('CBF'); plt.ylabel('ATT'); plt.savefig(os.path.join(FIG_DIR, "local_geometry_cosine.png")); plt.close()

print("Running Experiment E (Noise-aware Fisher Analysis with Empirical Covariance)...")
snr_list = [5, 10, 20, 30, 50, 100, np.inf]
fisher_records = []
rng = np.random.default_rng(42)

# Subsample for Fisher calculation to save time
cbf_fish = np.linspace(20, 80, 20)
att_fish = np.linspace(0.5, 3.0, 20)
C_fish, A_fish = np.meshgrid(cbf_fish, att_fish)
C_fish_flat, A_fish_flat = C_fish.flatten(), A_fish.flatten()
S_fish = compute_signals_vec(C_fish_flat, A_fish_flat) * SCALE
J_cbf_fish, J_att_fish = analytical_jacobian(C_fish_flat, A_fish_flat)

N_MC = 5000  # Monte carlo samples for covariance estimation

for snr in snr_list:
    noise_sd = calculate_noise_sd_from_snr(snr)
    for i in range(len(C_fish_flat)):
        s_base = S_fish[i:i+1] # shape (1, 4)
        J = np.column_stack([J_cbf_fish[i] * std_CBF, J_att_fish[i] * std_ATT])
        
        if np.isinf(snr) or noise_sd == 0:
            s_cbf_phys, s_att_phys = 0.0, 0.0
        else:
            # Empirical Covariance Estimation
            s_rep = np.repeat(s_base, N_MC, axis=0)
            X_noisy = add_noise(s_rep, noise_sd, rng)
            Sigma = np.cov(X_noisy, rowvar=False) + np.eye(4)*1e-6 # small regularization
            Sigma_inv = np.linalg.inv(Sigma)
            
            F = J.T @ Sigma_inv @ J
            try:
                Cov = np.linalg.inv(F)
                s_cbf_phys = np.sqrt(Cov[0,0]) * std_CBF if Cov[0,0] > 0 else np.nan
                s_att_phys = np.sqrt(Cov[1,1]) * std_ATT if Cov[1,1] > 0 else np.nan
            except:
                s_cbf_phys, s_att_phys = np.nan, np.nan
                
        fisher_records.append({
            'SNR': snr, 'CBF': C_fish_flat[i], 'ATT': A_fish_flat[i],
            'sigma_CBF': s_cbf_phys, 'sigma_ATT': s_att_phys
        })

fisher_df = pd.DataFrame(fisher_records)
fisher_df.to_csv(os.path.join(TAB_DIR, "fisher_information.csv"), index=False)

print("Running Experiment F (Ambiguity Analysis across SNRs)...")
# For ambiguity, we calculate the pairwise signal distance and compare to multiple noise levels
ambig_records = []
sub_indices = np.random.choice(len(C_dense_flat), 500, replace=False)
for i in range(len(sub_indices)):
    idx1 = sub_indices[i]
    for j in range(i+1, len(sub_indices)):
        idx2 = sub_indices[j]
        cbf1, att1 = C_dense_flat[idx1], A_dense_flat[idx1]
        cbf2, att2 = C_dense_flat[idx2], A_dense_flat[idx2]
        
        sig_dist = np.linalg.norm(S_dense[idx1] - S_dense[idx2])
        param_dist = np.sqrt(((cbf1 - cbf2)/std_CBF)**2 + ((att1 - att2)/std_ATT)**2)
        
        if param_dist > 0.5: # Only look at pairs that are far apart in parameter space
            ambig_records.append({
                'CBF_a': cbf1, 'ATT_a': att1, 'CBF_b': cbf2, 'ATT_b': att2,
                'signal_distance': sig_dist,
                'normalized_parameter_distance': param_dist
            })

ambig_df = pd.DataFrame(ambig_records)
ambig_df.to_csv(os.path.join(TAB_DIR, "ambiguity_analysis.csv"), index=False)

snr_thresholds = [5, 10, 20, 50]
print("\nAmbiguity Distribution:")
for snr in snr_thresholds:
    nsd = calculate_noise_sd_from_snr(snr)
    # Count pairs whose signal distance is within 1 std of the noise
    ambiguous_pairs = len(ambig_df[ambig_df['signal_distance'] <= nsd])
    total_pairs = len(ambig_df)
    print(f"SNR {snr:2d} (Noise SD {nsd:.2f}): {ambiguous_pairs} ambiguous pairs (out of {total_pairs} distant pairs)")

print("\nAll validation experiments completed successfully.")
