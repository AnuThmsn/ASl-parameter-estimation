import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization

# Dynamic imports
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_02 = load_module("mod_02", "architectures/02_hierarchical_kinetic_conditioning/model.py")
mod_03 = load_module("mod_03", "architectures/03_gradient_isolated_kinetic_conditioning/model.py")
mod_04 = load_module("mod_04", "architectures/04_partial_gradient_kinetic_conditioning/model.py")
mod_05 = load_module("mod_05", "architectures/05_adaptive_gradient_kinetic_conditioning/model.py")
mod_fx = load_module("mod_fx", "experiments/gradient_coupling_study/model_fixed.py")

def build_model(arch_name, alpha=None):
    if arch_name == "arch02": return mod_02.HierarchicalKineticNet()
    if arch_name == "arch03": return mod_03.GradientIsolatedKineticNet()
    if arch_name == "arch04": return mod_04.PartialGradientKineticNet(alpha=0.5)
    if arch_name == "arch05": return mod_05.AdaptiveGradientKineticNet()
    if arch_name == "fixed_alpha": return mod_fx.FixedGradientKineticNet(alpha=alpha)
    raise ValueError(f"Unknown model: {arch_name}")

def train_model(model, train_loader, val_loader, config, save_path, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = nn.L1Loss()
    
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    grad_clip = config['training']['grad_clip']
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_weights = None
    p_counter = 0
    t0 = time.time()
    
    history = []
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            pred = out[0] # Handle models returning 2 or 3 values
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
        tr_loss /= len(train_loader.dataset)
        
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                pred = out[0]
                loss = criterion(pred, by)
                va_loss += loss.item() * bx.size(0)
        va_loss /= len(val_loader.dataset)
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_counter = 0
        else:
            p_counter += 1
            
        history.append({
            'epoch': ep, 'train_loss': tr_loss, 'val_loss': va_loss,
            'best_val_loss': best_val_loss, 'best_epoch': best_epoch,
            'elapsed_time': time.time() - t0
        })
        
        if p_counter >= patience:
            print(f"    Early stopping at epoch {ep} (best: {best_epoch} with val_loss: {best_val_loss:.5f})", flush=True)
            break
            
    if best_weights is not None:
        torch.save(best_weights, save_path)
    
    return history, best_epoch, best_val_loss, history[30]['val_loss'] if len(history) > 30 else history[-1]['val_loss'], history[-1]['val_loss']

def main():
    with open("final_training_audit/configs/final_training_config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Starting Final Training Audit ({device}) ===", flush=True)
    
    print("Generating fixed datasets...", flush=True)
    rng_tr = np.random.default_rng(cfg['data']['train_seed'])
    X_tr, y_tr, _ = generate_data(cfg['data']['n_train'], rng_tr, cfg['data']['n_noise_levels'], cfg['data']['sd_max'])
    
    rng_va = np.random.default_rng(cfg['data']['val_seed'])
    X_va, y_va, _ = generate_data(cfg['data']['n_val'], rng_va, cfg['data']['n_noise_levels'], cfg['data']['sd_max'])
    
    x_mean, x_std = fit_normalization(X_tr)
    X_tr_norm = apply_normalization(X_tr, x_mean, x_std)
    X_va_norm = apply_normalization(X_va, x_mean, x_std)
    
    y_mean, y_std = fit_target_standardization(y_tr)
    y_tr_norm = apply_target_standardization(y_tr, y_mean, y_std)
    y_va_norm = apply_target_standardization(y_va, y_mean, y_std)
    
    np.savez("final_training_audit/configs/canonical_normalization.npz", 
             x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std)
    print("Saved canonical normalization from train set.")
    
    train_ds = TensorDataset(torch.tensor(X_tr_norm, dtype=torch.float32), torch.tensor(y_tr_norm, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va_norm, dtype=torch.float32), torch.tensor(y_va_norm, dtype=torch.float32))
    
    bs = cfg['training']['batch_size']
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    tasks = []
    # Skip models that are already trained successfully
    import glob
    existing = set([os.path.basename(p).replace(".pt", "") for p in glob.glob("final_training_audit/checkpoints/**/*.pt", recursive=True)])
    
    for arch in ["arch02", "arch03", "arch04", "arch05"]:
        for seed in cfg['training']['seeds']:
            name = f"{arch}_seed{seed}"
            if name not in existing:
                tasks.append((arch, seed, None))
            
    for alpha in cfg['fixed_alphas']:
        for seed in cfg['training']['seeds']:
            name = f"fixed_alpha_a{alpha:.2f}_seed{seed}"
            if name not in existing:
                tasks.append(("fixed_alpha", seed, alpha))
            
    # Load convergence records if it exists
    convergence_records = []
    if os.path.exists("final_training_audit/audit/convergence_summary.csv"):
        convergence_records = pd.read_csv("final_training_audit/audit/convergence_summary.csv").to_dict('records')
    
    for arch, seed, alpha in tasks:
        task_name = f"{arch}_seed{seed}" if alpha is None else f"{arch}_a{alpha:.2f}_seed{seed}"
        print(f"\nTraining {task_name}...", flush=True)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = build_model(arch, alpha)
        model.x_mean = torch.tensor(x_mean, dtype=torch.float32).to(device)
        model.x_std = torch.tensor(x_std, dtype=torch.float32).to(device)
        model.y_mean = torch.tensor(y_mean, dtype=torch.float32).to(device)
        model.y_std = torch.tensor(y_std, dtype=torch.float32).to(device)
        
        ckpt_dir = f"final_training_audit/checkpoints/{arch}" if alpha is None else f"final_training_audit/checkpoints/fixed_alpha"
        hist_dir = f"final_training_audit/histories/{arch}" if alpha is None else f"final_training_audit/histories/fixed_alpha"
        
        save_path = f"{ckpt_dir}/{task_name}.pt"
        
        hist, best_ep, best_val, val_at_30, val_final = train_model(model, train_dl, val_dl, cfg, save_path, device)
        
        df_hist = pd.DataFrame(hist)
        df_hist.to_csv(f"{hist_dir}/{task_name}.csv", index=False)
        
        convergence_records.append({
            "Model": f"{arch}_a{alpha:.2f}" if alpha is not None else arch,
            "Seed": seed,
            "Best epoch": best_ep,
            "Best validation loss": best_val,
            "Loss at epoch 30": val_at_30,
            "Final epoch loss": val_final,
            "Improvement after epoch 30": val_at_30 - best_val if best_ep > 30 else 0.0,
            "Stopped early?": len(hist) < cfg['training']['epochs'],
            "Reason for stopping": "Patience reached" if len(hist) < cfg['training']['epochs'] else "Max epochs"
        })
        
        # Save incrementally
        pd.DataFrame(convergence_records).to_csv("final_training_audit/audit/convergence_summary.csv", index=False)
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
