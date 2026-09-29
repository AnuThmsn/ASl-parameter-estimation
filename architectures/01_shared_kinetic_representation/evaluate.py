import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization
from src.physics.asl_forward import compute_signals_vec, SCALE, PLDs
from src.training.evaluation import calc_metrics
from model import SharedKineticNet
from train import BaselineComboNet, get_data

def evaluate():
    rng = np.random.default_rng(42)
    # We just need a large clean test set to do range-based evaluations
    _, _, _, X_mean, X_std, Y_mean, Y_std = get_data(rng, n_train=1000, n_val=1000, n_test=100) # Dummy for normalizers
    
    # Load normalizers from a fixed seed data gen
    X_tr, Y_tr, _ = generate_data(20_000, np.random.default_rng(42))
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    
    y_min, y_max = [0.0, 0.5], [100.0, 3.0]
    
    # Load Arch01 Seed 42
    model = SharedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=y_min, y_max=y_max)
    model.load_state_dict(torch.load("architectures/01_shared_kinetic_representation/checkpoints/Architecture01_seed42.pt", weights_only=True))
    model.eval()
    
    baseline = BaselineComboNet(Y_mean, Y_std, y_min, y_max)
    baseline.load_state_dict(torch.load("architectures/01_shared_kinetic_representation/checkpoints/Baseline_seed42.pt", weights_only=True))
    baseline.eval()

    # Generate a dense clean test set for diagnostic analysis
    X_ts_clean, Y_ts_clean, _ = generate_data(100_000, np.random.default_rng(99), n_noise_levels=0)
    X_ts_norm = apply_normalization(X_ts_clean, X_mean, X_std)
    
    # Predict
    phys_pred_arch01 = model.predict_physical(torch.tensor(X_ts_norm)).numpy()
    phys_pred_base = baseline.predict_physical(torch.tensor(X_ts_norm)).numpy()
    
    # ---------------------------------------------------------
    # PART 10: ATT and CBF Range Errors
    # ---------------------------------------------------------
    att_ranges = [(0.5, 1.0), (1.0, 1.4), (1.4, 1.525), (1.525, 1.8), (1.8, 2.2), (2.2, 2.6), (2.6, 3.0)]
    cbf_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    
    att_results = []
    cbf_results = []
    
    true_cbf = Y_ts_clean[:, 0]
    true_att = Y_ts_clean[:, 1]
    
    for (low, high) in att_ranges:
        mask = (true_att >= low) & (true_att < high)
        if mask.sum() > 0:
            m = calc_metrics(true_att[mask], phys_pred_arch01[mask, 1])
            m_cbf = calc_metrics(true_cbf[mask], phys_pred_arch01[mask, 0])
            att_results.append({'Range': f"{low}-{high}", 'Count': mask.sum(), 'ATT_RMSE': m['RMSE'], 'CBF_RMSE': m_cbf['RMSE']})
            
    for (low, high) in cbf_ranges:
        mask = (true_cbf >= low) & (true_cbf < high)
        if mask.sum() > 0:
            m = calc_metrics(true_cbf[mask], phys_pred_arch01[mask, 0])
            m_att = calc_metrics(true_att[mask], phys_pred_arch01[mask, 1])
            cbf_results.append({'Range': f"{low}-{high}", 'Count': mask.sum(), 'CBF_RMSE': m['RMSE'], 'ATT_RMSE': m_att['RMSE']})
            
    pd.DataFrame(att_results).to_csv("architectures/01_shared_kinetic_representation/tables/error_by_att_range.csv", index=False)
    pd.DataFrame(cbf_results).to_csv("architectures/01_shared_kinetic_representation/tables/error_by_cbf_range.csv", index=False)

    # ---------------------------------------------------------
    # PART 12: Generalization / Sanity Tests
    # ---------------------------------------------------------
    # 1. CBF monotonicity (Fixed ATT=1.5)
    cbf_grid = np.linspace(0, 100, 100)
    att_grid = np.full_like(cbf_grid, 1.5)
    S_cbf = compute_signals_vec(cbf_grid, att_grid) * SCALE
    S_cbf_norm = apply_normalization(S_cbf, X_mean, X_std)
    pred_cbf_mono = model.predict_physical(torch.tensor(S_cbf_norm).float()).numpy()[:, 0]
    
    plt.figure(figsize=(6,4)); plt.plot(cbf_grid, pred_cbf_mono); plt.plot([0,100], [0,100], 'k--')
    plt.xlabel('True CBF'); plt.ylabel('Predicted CBF'); plt.title('CBF Monotonicity (ATT=1.5)')
    plt.savefig("architectures/01_shared_kinetic_representation/figures/cbf_monotonicity.png"); plt.close()

    # 2. ATT response (Fixed CBF=50)
    att_grid = np.linspace(0.5, 3.0, 100)
    cbf_grid = np.full_like(att_grid, 50.0)
    S_att = compute_signals_vec(cbf_grid, att_grid) * SCALE
    S_att_norm = apply_normalization(S_att, X_mean, X_std)
    pred_att_resp = model.predict_physical(torch.tensor(S_att_norm).float()).numpy()[:, 1]
    
    plt.figure(figsize=(6,4)); plt.plot(att_grid, pred_att_resp); plt.plot([0.5,3.0], [0.5,3.0], 'k--')
    for pld in PLDs: plt.axvline(pld, color='r', linestyle=':', alpha=0.5)
    plt.xlabel('True ATT'); plt.ylabel('Predicted ATT'); plt.title('ATT Response (CBF=50)')
    plt.savefig("architectures/01_shared_kinetic_representation/figures/att_response.png"); plt.close()

    # 3. Clean signal reconstruction
    S_recon_arch01 = compute_signals_vec(phys_pred_arch01[:, 0], phys_pred_arch01[:, 1]) * SCALE
    S_recon_base = compute_signals_vec(phys_pred_base[:, 0], phys_pred_base[:, 1]) * SCALE
    
    recon_err_arch01 = np.mean(np.linalg.norm(S_recon_arch01 - X_ts_clean, axis=1))
    recon_err_base = np.mean(np.linalg.norm(S_recon_base - X_ts_clean, axis=1))
    print(f"Reconstruction Error (Arch01): {recon_err_arch01:.4f}")
    print(f"Reconstruction Error (Baseline): {recon_err_base:.4f}")

    # ---------------------------------------------------------
    # PART 13: Representation Analysis
    # ---------------------------------------------------------
    # Subset of 2000 points for PCA
    sub_mask = np.random.choice(len(X_ts_norm), 2000, replace=False)
    X_sub = torch.tensor(X_ts_norm[sub_mask]).float()
    with torch.no_grad():
        _, latent = model(X_sub)
    latent_np = latent.numpy()
    
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_np)
    
    true_cbf_sub = true_cbf[sub_mask]
    true_att_sub = true_att[sub_mask]
    
    plt.figure(figsize=(8,6)); sc = plt.scatter(latent_pca[:,0], latent_pca[:,1], c=true_cbf_sub, cmap='viridis', s=10)
    plt.colorbar(sc, label='True CBF'); plt.title('Shared Latent Space (PCA) - CBF')
    plt.savefig("architectures/01_shared_kinetic_representation/figures/shared_latent_pca_cbf.png"); plt.close()

    plt.figure(figsize=(8,6)); sc = plt.scatter(latent_pca[:,0], latent_pca[:,1], c=true_att_sub, cmap='plasma', s=10)
    plt.colorbar(sc, label='True ATT'); plt.title('Shared Latent Space (PCA) - ATT')
    plt.savefig("architectures/01_shared_kinetic_representation/figures/shared_latent_pca_att.png"); plt.close()

    # Correlations
    cbf_corrs = [np.corrcoef(latent_np[:, i], true_cbf_sub)[0, 1] for i in range(latent_np.shape[1])]
    att_corrs = [np.corrcoef(latent_np[:, i], true_att_sub)[0, 1] for i in range(latent_np.shape[1])]
    print(f"Max absolute CBF correlation in latent dims: {np.max(np.abs(cbf_corrs)):.4f}")
    print(f"Max absolute ATT correlation in latent dims: {np.max(np.abs(att_corrs)):.4f}")
    
if __name__ == "__main__":
    evaluate()
    print("Evaluation completed successfully.")
