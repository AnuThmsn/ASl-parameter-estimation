import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.generate_dataset import generate_data
from src.data.normalization import fit_normalization, apply_normalization, fit_target_standardization, apply_target_standardization
from src.models.common import StandardizedNet
from src.training.evaluation import calc_metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'architectures', '01_shared_kinetic_representation'))
from model import SharedKineticNet

class BaselineComboNet(nn.Module):
    def __init__(self, y_mean, y_std, y_min, y_max):
        super().__init__()
        # CBFnet: 9 layers x 50
        self.cbf_net = StandardizedNet(4, 9, 50, y_mean[0], y_std[0], y_min[0], y_max[0])
        # ATTnet: 9 layers x 100
        self.att_net = StandardizedNet(4, 9, 100, y_mean[1], y_std[1], y_min[1], y_max[1])
        self.y_std = torch.tensor(y_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_min = y_min
        self.y_max = y_max

    def forward(self, x):
        cbf_std = self.cbf_net(x).squeeze(-1)
        att_std = self.att_net(x).squeeze(-1)
        return torch.stack([cbf_std, att_std], dim=-1)

    def predict_physical(self, x):
        with torch.no_grad():
            std_pred = self.forward(x)
        phys_pred = std_pred * self.y_std.to(x.device) + self.y_mean.to(x.device)
        phys_pred[:, 0] = phys_pred[:, 0].clamp(self.y_min[0], self.y_max[0])
        phys_pred[:, 1] = phys_pred[:, 1].clamp(self.y_min[1], self.y_max[1])
        return phys_pred

class AblationIndependentNet(nn.Module):
    """Same capacity as SharedKineticNet but entirely independent (no sharing)"""
    def __init__(self, y_mean, y_std, y_min, y_max):
        super().__init__()
        # 6 layers x 100 + 3 layers x 50 (approx 60k parameters)
        self.cbf_net = StandardizedNet(4, 9, 70, y_mean[0], y_std[0], y_min[0], y_max[0]) # Adjusted width to match ~45k params
        self.att_net = StandardizedNet(4, 9, 70, y_mean[1], y_std[1], y_min[1], y_max[1])
        self.y_std = torch.tensor(y_std, dtype=torch.float32)
        self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
        self.y_min = y_min
        self.y_max = y_max

    def forward(self, x):
        cbf_std = self.cbf_net(x).squeeze(-1)
        att_std = self.att_net(x).squeeze(-1)
        return torch.stack([cbf_std, att_std], dim=-1)

    def predict_physical(self, x):
        with torch.no_grad():
            std_pred = self.forward(x)
        phys_pred = std_pred * self.y_std.to(x.device) + self.y_mean.to(x.device)
        phys_pred[:, 0] = phys_pred[:, 0].clamp(self.y_min[0], self.y_max[0])
        phys_pred[:, 1] = phys_pred[:, 1].clamp(self.y_min[1], self.y_max[1])
        return phys_pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_data(rng, n_train=20_000, n_val=5_000, n_test=5_000):
    print("Generating datasets...")
    X_tr, Y_tr, _ = generate_data(n_train, rng)
    X_va, Y_va, _ = generate_data(n_val, rng)
    
    # Test set evaluated at fixed SNRs
    test_data = {}
    for snr in [np.inf, 50, 20, 15, 10, 5]:
        rng_test = np.random.default_rng(7)
        if np.isinf(snr):
            X_ts, Y_ts, _ = generate_data(n_test, rng_test, n_noise_levels=0)
        else:
            X_ts, Y_ts, _ = generate_data(n_test, rng_test, n_noise_levels=1, sd_max=334.2039/snr)
        test_data[snr] = (X_ts, Y_ts)
        
    X_mean, X_std = fit_normalization(X_tr)
    Y_mean, Y_std = fit_target_standardization(Y_tr)
    
    X_tr_norm = apply_normalization(X_tr, X_mean, X_std)
    X_va_norm = apply_normalization(X_va, X_mean, X_std)
    Y_tr_std = apply_target_standardization(Y_tr, Y_mean, Y_std)
    Y_va_std = apply_target_standardization(Y_va, Y_mean, Y_std)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_norm), torch.tensor(Y_tr_std)), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_va_norm), torch.tensor(Y_va_std)), batch_size=1024, shuffle=False)
    
    return train_loader, val_loader, test_data, X_mean, X_std, Y_mean, Y_std


def train_model(model, train_loader, val_loader, epochs=100, patience=15, lr=1e-3):
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
            pred = model(bx)
            if isinstance(pred, tuple):
                pred = pred[0] # handle shared returning latent
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                if isinstance(pred, tuple): pred = pred[0]
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
    
if __name__ == "__main__":
    seeds = [42, 123, 2024]
    results = []
    
    for seed in seeds:
        print(f"\\n--- RUNNING SEED {seed} ---")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        train_loader, val_loader, test_data, X_mean, X_std, Y_mean, Y_std = get_data(rng)
        y_min, y_max = [0.0, 0.5], [100.0, 3.0]
        
        models_to_train = {
            "Baseline": BaselineComboNet(Y_mean, Y_std, y_min, y_max),
            "Architecture01": SharedKineticNet(y_mean=Y_mean, y_std=Y_std, y_min=y_min, y_max=y_max),
            "Ablation": AblationIndependentNet(Y_mean, Y_std, y_min, y_max)
        }
        
        for name, model in models_to_train.items():
            print(f"\\nTraining {name} (Params: {count_parameters(model)})...")
            best_val, ep_count, time_taken = train_model(model, train_loader, val_loader, epochs=30, patience=10)
            print(f"Finished in {ep_count} epochs, time: {time_taken:.1f}s, Best Val Loss: {best_val:.4f}")
            
            torch.save(model.state_dict(), f"architectures/01_shared_kinetic_representation/checkpoints/{name}_seed{seed}.pt")
            
            model.eval()
            for snr, (X_ts, Y_ts) in test_data.items():
                X_ts_norm = apply_normalization(X_ts, X_mean, X_std)
                phys_pred = model.predict_physical(torch.tensor(X_ts_norm)).numpy()
                
                cbf_metrics = calc_metrics(Y_ts[:, 0], phys_pred[:, 0])
                att_metrics = calc_metrics(Y_ts[:, 1], phys_pred[:, 1])
                
                results.append({
                    'Seed': seed,
                    'Model': name,
                    'SNR': snr,
                    'CBF RMSE': cbf_metrics['RMSE'],
                    'CBF MAE': cbf_metrics['MAE'],
                    'CBF R2': cbf_metrics['R2'],
                    'CBF CCC': cbf_metrics['CCC'],
                    'ATT RMSE': att_metrics['RMSE'],
                    'ATT MAE': att_metrics['MAE'],
                    'ATT R2': att_metrics['R2'],
                    'ATT CCC': att_metrics['CCC']
                })
                
    pd.DataFrame(results).to_csv("architectures/01_shared_kinetic_representation/tables/raw_results.csv", index=False)
    
    # Aggregate results across seeds
    df = pd.DataFrame(results)
    agg_df = df.groupby(['Model', 'SNR']).mean().drop(columns=['Seed']).reset_index()
    agg_df.to_csv("comparisons/architecture_00_vs_01.csv", index=False)
    print("Done! Results saved.")
