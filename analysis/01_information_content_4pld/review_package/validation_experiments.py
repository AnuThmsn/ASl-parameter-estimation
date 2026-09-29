import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.asl_forward import compute_signals_vec, PLDs, tau, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX, lmbda, alpha, beta, T1t, T1a
from src.physics.noise_models import calculate_noise_sd_from_snr

OUT_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
REV_DIR = os.path.join(OUT_DIR, "review_package")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# Standardized parameters mapping
mean_CBF = 50.0
std_CBF = 100.0 / np.sqrt(12)
mean_ATT = 1.75
std_ATT = 2.5 / np.sqrt(12)

# --- EXPERIMENT B: Analytical Derivatives ---
def analytical_jacobian(cbf, att):
    """Computes exact analytical Jacobian for a vector of CBF and ATT."""
    # S = prefix * e_att * (t1 - t2)
    # prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * (CBF / (6000.0 * lmbda))
    cbf_expanded = cbf[:, None]
    att_expanded = att[:, None]
    plds = PLDs[None, :]
    
    # 1. dS/dCBF is simply S / CBF
    S = compute_signals_vec(cbf, att) * SCALE
    # Avoid division by zero
    dS_dCBF = np.zeros_like(S)
    mask = cbf_expanded[:, 0] > 1e-12
    dS_dCBF[mask] = S[mask] / cbf_expanded[mask]
    
    # 2. dS/dATT
    # S = K * exp(-ATT/T1a) * (exp(-max(PLD-ATT,0)/T1t) - exp(-max(PLD+tau-ATT,0)/T1t))
    # Let K' = K * exp(-ATT/T1a).
    # dS/dATT = (-1/T1a) * S + K' * d(term)/dATT
    f_per_s = cbf_expanded / (6000.0 * lmbda)
    prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s * SCALE
    e_att = np.exp(-att_expanded / T1a)
    K_prime = prefix * e_att
    
    plds_exp = np.broadcast_to(plds, S.shape)
    
    # piece 1: max(PLD-ATT, 0)
    term1_deriv = np.zeros_like(S)
    m1 = att_expanded < plds_exp
    term1_deriv[m1] = (1.0 / T1t) * np.exp(-(plds_exp[m1] - att_expanded.repeat(4, axis=1)[m1]) / T1t)
    
    # piece 2: max(PLD+tau-ATT, 0)
    term2_deriv = np.zeros_like(S)
    m2 = att_expanded < (plds_exp + tau)
    term2_deriv[m2] = (1.0 / T1t) * np.exp(-(plds_exp[m2] + tau - att_expanded.repeat(4, axis=1)[m2]) / T1t)
    
    dS_dATT = (-1.0 / T1a) * S + K_prime * (term1_deriv - term2_deriv)
    
    return dS_dCBF, dS_dATT

# --- EXPERIMENT A: Finite Difference Stability ---
print("Running Experiment A & B...")
cbf_pts = [20.0, 50.0, 80.0]
att_pts = [0.5, 1.0, 1.5, 1.6, 2.0, 2.1, 2.5, 2.6, 3.0]
cbf_grid, att_grid = np.meshgrid(cbf_pts, att_pts)
cbf_flat = cbf_grid.flatten()
att_flat = att_grid.flatten()

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
            # Relative error against analytical
            err_cbf = np.linalg.norm(J_num_cbf[i] - J_ana_cbf[i]) / (np.linalg.norm(J_ana_cbf[i]) + 1e-12)
            err_att = np.linalg.norm(J_num_att[i] - J_ana_att[i]) / (np.linalg.norm(J_ana_att[i]) + 1e-12)
            
            # Cond number & Correlation using numerical J
            J_num = np.column_stack([J_num_cbf[i] * std_CBF, J_num_att[i] * std_ATT]) # standardized
            U, S, Vh = np.linalg.svd(J_num)
            cond = S[0] / S[1] if S[1] > 1e-12 else np.inf
            n1, n2 = np.linalg.norm(J_num[:,0]), np.linalg.norm(J_num[:,1])
            corr = np.dot(J_num[:,0], J_num[:,1]) / (n1 * n2) if n1>0 and n2>0 else 0
            
            fd_records.append({
                'CBF': cbf_flat[i], 'ATT': att_flat[i],
                'h_CBF': h_cbf, 'h_ATT': h_att,
                'err_CBF': err_cbf, 'err_ATT': err_att,
                'cond_num': cond, 'correlation': corr
            })
pd.DataFrame(fd_records).to_csv(os.path.join(TAB_DIR, "finite_difference_stability.csv"), index=False)

# Plot stability for a specific point (CBF=50, ATT=1.6)
point_df = pd.DataFrame(fd_records)
point_df = point_df[(point_df['CBF']==50.0) & (point_df['ATT']==1.6)]
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.scatter(point_df['h_CBF'], point_df['err_CBF'])
plt.xlabel('h_CBF')
plt.ylabel('Relative Error J_CBF')
plt.title('FD Stability CBF (50, 1.6)')
plt.subplot(122)
plt.scatter(point_df['h_ATT'], point_df['err_ATT'])
plt.xlabel('h_ATT')
plt.ylabel('Relative Error J_ATT')
plt.title('FD Stability ATT (50, 1.6)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "finite_difference_stability_50_1.6.png"))
plt.close()

# --- EXPERIMENT C: Signal Manifold Geometry ---
print("Running Experiment C...")
cbf_dense = np.linspace(10, 100, 50)
att_dense = np.linspace(0.5, 3.0, 50)
C_dense, A_dense = np.meshgrid(cbf_dense, att_dense)
C_dense_flat, A_dense_flat = C_dense.flatten(), A_dense.flatten()
S_dense = compute_signals_vec(C_dense_flat, A_dense_flat) * SCALE

pd.DataFrame({
    'CBF': C_dense_flat, 'ATT': A_dense_flat,
    'S1': S_dense[:,0], 'S2': S_dense[:,1], 'S3': S_dense[:,2], 'S4': S_dense[:,3]
}).to_csv(os.path.join(REV_DIR, "parameter_signal_grid.csv"), index=False)

pca = PCA(n_components=2)
S_pca = pca.fit_transform(S_dense)

plt.figure(figsize=(8,6))
sc = plt.scatter(S_pca[:,0], S_pca[:,1], c=C_dense_flat, cmap='viridis', s=10)
plt.colorbar(sc, label='CBF')
plt.title('Signal Manifold (PCA) colored by CBF')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(os.path.join(FIG_DIR, "signal_manifold_pca_cbf.png"))
plt.close()

plt.figure(figsize=(8,6))
sc = plt.scatter(S_pca[:,0], S_pca[:,1], c=A_dense_flat, cmap='plasma', s=10)
plt.colorbar(sc, label='ATT')
plt.title('Signal Manifold (PCA) colored by ATT')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(os.path.join(FIG_DIR, "signal_manifold_pca_att.png"))
plt.close()

# Pairwise plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(S_dense[:,0], S_dense[:,1], c=A_dense_flat, cmap='plasma', s=5)
axes[0].set_xlabel('S1'); axes[0].set_ylabel('S2')
axes[1].scatter(S_dense[:,1], S_dense[:,2], c=A_dense_flat, cmap='plasma', s=5)
axes[1].set_xlabel('S2'); axes[1].set_ylabel('S3')
sc = axes[2].scatter(S_dense[:,2], S_dense[:,3], c=A_dense_flat, cmap='plasma', s=5)
axes[2].set_xlabel('S3'); axes[2].set_ylabel('S4')
fig.colorbar(sc, ax=axes.ravel().tolist(), label='ATT')
plt.suptitle('Pairwise Signal Plots')
plt.savefig(os.path.join(FIG_DIR, "pairwise_signals.png"))
plt.close()

# --- EXPERIMENT D: Local CBF/ATT Geometry ---
# Cosine similarity uses analytical jacobian
print("Running Experiment D...")
cos_theta = np.zeros(len(C_dense_flat))
J_ana_cbf_dense, J_ana_att_dense = analytical_jacobian(C_dense_flat, A_dense_flat)

for i in range(len(C_dense_flat)):
    n1 = np.linalg.norm(J_ana_cbf_dense[i])
    n2 = np.linalg.norm(J_ana_att_dense[i])
    if n1 > 0 and n2 > 0:
        cos_theta[i] = np.dot(J_ana_cbf_dense[i], J_ana_att_dense[i]) / (n1 * n2)

plt.figure(figsize=(8,6))
plt.imshow(np.abs(cos_theta.reshape(C_dense.shape)), origin='lower',
           extent=[cbf_dense[0], cbf_dense[-1], att_dense[0], att_dense[-1]], aspect='auto', cmap='coolwarm')
plt.colorbar(label='|cos(theta)|')
plt.title('Cosine Similarity between CBF and ATT directions')
plt.xlabel('CBF')
plt.ylabel('ATT')
plt.savefig(os.path.join(FIG_DIR, "local_geometry_cosine.png"))
plt.close()

# --- EXPERIMENT E: Noise-aware Information Analysis ---
print("Running Experiment E...")
snr_list = [5, 10, 20, 30, 50, 100, np.inf]
fisher_records = []

for snr in snr_list:
    noise_sd = calculate_noise_sd_from_snr(snr)
    # the noise model adds independent N(0, sd) to the complex parts, 
    # but the magnitude operation X = sqrt((s+e1)^2+e2^2) + ... is a Rice distribution.
    # For moderate/high SNR, Rice variance is approximately sigma^2 + sigma^2 = 2*sigma^2
    # Let's use var = 2 * noise_sd^2 as a diagonal covariance approximation for Fisher.
    var = 2.0 * noise_sd**2 if noise_sd > 0 else 0.0
    
    if var > 0:
        Sigma_inv = np.eye(4) / var
    else:
        Sigma_inv = None # Noiseless case handled via pseudo-inverse or infinite precision
        
    for i in range(len(C_dense_flat)):
        J = np.column_stack([J_ana_cbf_dense[i] * std_CBF, J_ana_att_dense[i] * std_ATT])
        if Sigma_inv is not None:
            F = J.T @ Sigma_inv @ J
            try:
                Cov = np.linalg.inv(F)
                s_cbf = np.sqrt(Cov[0,0])
                s_att = np.sqrt(Cov[1,1])
                # physical parameter uncertainty = standardized * std_param
                # wait, since J is wrt standardized params, Cov is Cov(z).
                # physical std = sqrt(Cov_z) * std_param.
                s_cbf_phys = s_cbf * std_CBF
                s_att_phys = s_att * std_ATT
            except:
                s_cbf_phys = np.nan
                s_att_phys = np.nan
        else:
            s_cbf_phys = 0.0
            s_att_phys = 0.0
            
        fisher_records.append({
            'SNR': snr, 'CBF': C_dense_flat[i], 'ATT': A_dense_flat[i],
            'sigma_CBF': s_cbf_phys, 'sigma_ATT': s_att_phys
        })

fisher_df = pd.DataFrame(fisher_records)
fisher_df.to_csv(os.path.join(REV_DIR, "fisher_information.csv"), index=False)

# Plot sigma_ATT at SNR=10
fisher_snr10 = fisher_df[fisher_df['SNR'] == 10]
plt.figure(figsize=(8,6))
plt.imshow(fisher_snr10['sigma_ATT'].values.reshape(C_dense.shape), origin='lower',
           extent=[cbf_dense[0], cbf_dense[-1], att_dense[0], att_dense[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='sigma_ATT (s)')
plt.title('ATT Uncertainty (SNR 10) from Fisher Info')
plt.xlabel('CBF')
plt.ylabel('ATT')
plt.savefig(os.path.join(FIG_DIR, "fisher_sigma_ATT_snr10.png"))
plt.close()

# --- EXPERIMENT F: Ambiguity Analysis ---
print("Running Experiment F...")
ambig_records = []
noise_sd_10 = calculate_noise_sd_from_snr(10)
# Use a subset to find ambiguities quickly
sub_indices = np.random.choice(len(C_dense_flat), 1000, replace=False)

for i in range(len(sub_indices)):
    idx1 = sub_indices[i]
    for j in range(i+1, len(sub_indices)):
        idx2 = sub_indices[j]
        cbf1, att1 = C_dense_flat[idx1], A_dense_flat[idx1]
        cbf2, att2 = C_dense_flat[idx2], A_dense_flat[idx2]
        
        sig_dist = np.linalg.norm(S_dense[idx1] - S_dense[idx2])
        param_dist = np.sqrt(((cbf1 - cbf2)/std_CBF)**2 + ((att1 - att2)/std_ATT)**2)
        
        # Log pairs where signals are close but parameters are far
        if sig_dist < noise_sd_10 * 2.0 and param_dist > 0.1:
            ambig_records.append({
                'CBF_a': cbf1, 'ATT_a': att1, 'CBF_b': cbf2, 'ATT_b': att2,
                'signal_distance': sig_dist,
                'CBF_difference': abs(cbf1 - cbf2),
                'ATT_difference': abs(att1 - att2),
                'normalized_parameter_distance': param_dist
            })

ambig_df = pd.DataFrame(ambig_records)
ambig_df.to_csv(os.path.join(REV_DIR, "ambiguity_analysis.csv"), index=False)

if len(ambig_df) > 0:
    plt.figure(figsize=(8,6))
    sc = plt.scatter(ambig_df['signal_distance'], ambig_df['normalized_parameter_distance'], alpha=0.5)
    plt.axvline(noise_sd_10, color='r', linestyle='--', label='Noise SD @ SNR 10')
    plt.xlabel('Signal Distance ||S_a - S_b||')
    plt.ylabel('Normalized Parameter Distance')
    plt.title('Ambiguity Analysis')
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "ambiguity_map.png"))
    plt.close()

# --- PART 9: CBF Scaling Analysis ---
print("Running CBF Scaling Analysis...")
cbf_test1 = 30.0
cbf_test2 = 80.0
S_cbf1 = compute_signals_vec(np.full_like(att_dense, cbf_test1), att_dense) * SCALE
S_cbf2 = compute_signals_vec(np.full_like(att_dense, cbf_test2), att_dense) * SCALE

ratio1 = S_cbf1 / cbf_test1
ratio2 = S_cbf2 / cbf_test2
diff_ratios = np.linalg.norm(ratio1 - ratio2, axis=1)

plt.figure(figsize=(8,6))
plt.plot(att_dense, diff_ratios)
plt.xlabel('ATT (s)')
plt.ylabel('Difference in S(CBF)/CBF')
plt.title('Linearity of CBF Scaling')
plt.savefig(os.path.join(FIG_DIR, "cbf_linearity.png"))
plt.close()

print("All validation experiments completed successfully.")
