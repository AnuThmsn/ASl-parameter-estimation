import os
import sys
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization
from experiments.gradient_coupling_study.model_fixed import FixedGradientKineticNet

def train_model(model, train_loader, val_loader, epochs=30, patience=15, lr=1e-3, grad_clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    start_time = time.time()
    
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred, _ = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred, _ = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict(best_weights)
    return best_val_loss, ep, time.time() - start_time

def get_data(seed):
    rng = np.random.default_rng(seed)
    # Canonical canonical configuration
    X_tr, Y_tr, _ = generate_data(20000, rng, n_noise_levels=100, sd_max=66.84078)
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    
    X_tr_norm = apply_normalization(X_tr, X_mean, X_std)
    Y_tr_norm = apply_target_standardization(Y_tr, Y_mean, Y_std)
    
    X_va, Y_va, _ = generate_data(5000, rng, n_noise_levels=100, sd_max=66.84078)
    X_va_norm = apply_normalization(X_va, X_mean, X_std)
    Y_va_norm = apply_target_standardization(Y_va, Y_mean, Y_std)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_norm).float(), torch.tensor(Y_tr_norm).float()), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va_norm).float(), torch.tensor(Y_va_norm).float()), batch_size=512, shuffle=False)
    
    return train_loader, val_loader, X_mean, X_std, Y_mean, Y_std

def main():
    seeds = [42, 123, 2024]
    alphas = [0.00, 0.25, 0.50, 0.75, 1.00]
    
    for seed in seeds:
        print(f"\n=== Preparing Data for Seed {seed} ===")
        train_loader, val_loader, X_m, X_s, Y_m, Y_s = get_data(seed)
        
        for alpha in alphas:
            ckpt_path = f"experiments/gradient_coupling_study/checkpoints/fixed_alpha_{alpha:.2f}_seed{seed}.pt"
            if os.path.exists(ckpt_path):
                print(f"Skipping Alpha {alpha:.2f} Seed {seed} (Already exists)")
                continue
                
            print(f"Training Alpha {alpha:.2f} | Seed {seed}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = FixedGradientKineticNet(y_mean=Y_m, y_std=Y_s, y_min=[0.0, 0.5], y_max=[100.0, 3.0], alpha=alpha)
            best_val, ep, t = train_model(model, train_loader, val_loader)
            print(f"  -> Finished in {ep} epochs, Time: {t:.1f}s, Val Loss: {best_val:.4f}")
            torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    main()
