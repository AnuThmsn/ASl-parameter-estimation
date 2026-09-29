import os
import sys
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_05 = load_module('model_05', 'architectures/05_adaptive_gradient_kinetic_conditioning/model.py')
AdaptiveGradientKineticNet = mod_05.AdaptiveGradientKineticNet

def train_model(model, train_loader, val_loader, epochs=100, patience=15, lr=1e-3, grad_clip=1.0):
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
            pred, _, _ = model(bx)
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
                pred, _, _ = model(bx)
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
                
    time_taken = time.time() - start_time
    model.load_state_dict(best_weights)
    return best_val_loss, ep, time_taken

def get_data(cfg, seed):
    rng = np.random.default_rng(seed)
    X_tr, Y_tr, _ = generate_data(cfg["data"]["n_train"], rng, n_noise_levels=cfg["data"]["n_noise_levels"], sd_max=cfg["data"]["sd_max"])
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    
    X_tr_norm = apply_normalization(X_tr, X_mean, X_std)
    Y_tr_norm = apply_target_standardization(Y_tr, Y_mean, Y_std)
    
    X_va, Y_va, _ = generate_data(cfg["data"]["n_val"], rng, n_noise_levels=cfg["data"]["n_noise_levels"], sd_max=cfg["data"]["sd_max"])
    X_va_norm = apply_normalization(X_va, X_mean, X_std)
    Y_va_norm = apply_target_standardization(Y_va, Y_mean, Y_std)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_norm).float(), torch.tensor(Y_tr_norm).float()), batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va_norm).float(), torch.tensor(Y_va_norm).float()), batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    return train_loader, val_loader, X_mean, X_std, Y_mean, Y_std

def main():
    with open("architectures/05_adaptive_gradient_kinetic_conditioning/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    y_min, y_max = cfg["data"]["y_min"], cfg["data"]["y_max"]
    os.makedirs("architectures/05_adaptive_gradient_kinetic_conditioning/checkpoints", exist_ok=True)
    os.makedirs("architectures/05_adaptive_gradient_kinetic_conditioning/tables", exist_ok=True)
    
    for seed in cfg["training"]["seeds"]:
        print(f"=== Starting seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_loader, val_loader, X_m, X_s, Y_m, Y_s = get_data(cfg, seed)
        
        model = AdaptiveGradientKineticNet(y_mean=Y_m, y_std=Y_s, y_min=y_min, y_max=y_max)
        print(f"Training AdaptiveGradientKineticNet (Seed {seed}) - Params: {sum(p.numel() for p in model.parameters())}")
        best_val, ep_count, time_taken = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            epochs=cfg["training"]["epochs"], patience=cfg["training"]["patience"],
            lr=cfg["training"]["lr"], grad_clip=cfg["training"]["grad_clip"]
        )
        print(f"Finished in {ep_count} epochs, time: {time_taken:.1f}s, Best Val Loss: {best_val:.4f}")
        torch.save(model.state_dict(), f"architectures/05_adaptive_gradient_kinetic_conditioning/checkpoints/Architecture05_seed{seed}.pt")

if __name__ == "__main__":
    main()
