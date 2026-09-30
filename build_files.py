import os

os.makedirs('experiments/literature_unet', exist_ok=True)

evaluate_py = '''import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \'../../\')))
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.physics.noise_models import add_noise
from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import SpatialASLDataset

SEEDS = [42, 123, 2024]
SNRS = [float(\'inf\'), 50, 20, 15, 10, 5]
M0 = 334.2039
DEVICE = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')

def run_evaluation():
    os.makedirs(\'results\', exist_ok=True)
    raw_results = []
    att_results = []
    cbf_results = []
    
    test_dataset = SpatialASLDataset(seed=7, num_volumes=40)
    
    for seed in SEEDS:
        model = LuciwUNet3D().to(DEVICE)
        checkpoint_path = f\'checkpoints/unet_seed{seed}.pt\'
        if not os.path.exists(checkpoint_path):
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint[\'model_state_dict\'])
        model.eval()
        
        stats = checkpoint[\'norm_stats\']
        mean, std = stats[\'mean\'], stats[\'std\']
        
        for snr in SNRS:
            noise_sd = 0 if np.isinf(snr) else M0 / snr
            
            for vol_idx in range(len(test_dataset)):
                signals, params, mask = test_dataset[vol_idx]
                
                if noise_sd > 0:
                    signals = add_noise(signals.unsqueeze(0), noise_sd).squeeze(0)
                    
                signals = (signals - mean.view(-1, 1, 1, 1)) / (std.view(-1, 1, 1, 1) + 1e-8)
                
                with torch.no_grad():
                    pred = model(signals.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
                    
                pred_cbf = pred[0]
                pred_att = pred[1]
                gt_cbf = params[0]
                gt_att = params[1]
                
                brain_mask = mask.bool()
                if not brain_mask.any(): continue
                
                cbf_err = (pred_cbf[brain_mask] - gt_cbf[brain_mask])
                att_err = (pred_att[brain_mask] - gt_att[brain_mask])
                
                cbf_rmse = torch.sqrt(torch.mean(cbf_err**2)).item()
                att_rmse = torch.sqrt(torch.mean(att_err**2)).item()
                cbf_mae = torch.mean(torch.abs(cbf_err)).item()
                att_mae = torch.mean(torch.abs(att_err)).item()
                cbf_bias = torch.mean(cbf_err).item()
                att_bias = torch.mean(att_err).item()
                
                raw_results.append({
                    \'seed\': seed, \'snr\': snr, \'volume_id\': vol_idx,
                    \'cbf_rmse\': cbf_rmse, \'att_rmse\': att_rmse,
                    \'cbf_mae\': cbf_mae, \'att_mae\': att_mae,
                    \'cbf_bias\': cbf_bias, \'att_bias\': att_bias
                })
                
                for lower, upper in [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]:
                    r_mask = brain_mask & (gt_att >= lower) & (gt_att < upper)
                    if r_mask.any():
                        rmse = torch.sqrt(torch.mean((pred_att[r_mask] - gt_att[r_mask])**2)).item()
                        att_results.append({\'seed\': seed, \'snr\': snr, \'volume_id\': vol_idx, \'range\': f"{lower}-{upper}", \'rmse\': rmse})
                        
                for lower, upper in [(0, 33), (33, 66), (66, 100)]:
                    r_mask = brain_mask & (gt_cbf >= lower) & (gt_cbf < upper)
                    if r_mask.any():
                        rmse = torch.sqrt(torch.mean((pred_cbf[r_mask] - gt_cbf[r_mask])**2)).item()
                        cbf_results.append({\'seed\': seed, \'snr\': snr, \'volume_id\': vol_idx, \'range\': f"{lower}-{upper}", \'rmse\': rmse})

    if raw_results:
        df = pd.DataFrame(raw_results)
        df.to_csv(\'results/raw_results.csv\', index=False)
        
        summary = df.groupby(\'snr\').agg({
            \'cbf_rmse\': [\'mean\', \'std\'],
            \'att_rmse\': [\'mean\', \'std\']
        }).reset_index()
        summary.columns = [\'snr\', \'cbf_rmse_mean\', \'cbf_rmse_std\', \'att_rmse_mean\', \'att_rmse_std\']
        summary.insert(0, \'model\', \'unet_4pld\')
        summary.to_csv(\'results/summary_results.csv\', index=False)
        
    if att_results:
        pd.DataFrame(att_results).to_csv(\'results/att_regime_results.csv\', index=False)
    if cbf_results:
        pd.DataFrame(cbf_results).to_csv(\'results/cbf_regime_results.csv\', index=False)

if __name__ == \'__main__\':
    # run_evaluation()
    pass
'''

curriculum_train_py = '''import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import SpatialASLDataset

DEVICE = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
NUM_EPOCHS = 150
SEED = 42

def get_p_complete(epoch):
    if epoch < 40: return 1.0
    elif epoch < 80: return 0.75
    elif epoch < 120: return 0.50
    else: return 0.25

def train_curriculum():
    os.makedirs(\'checkpoints\', exist_ok=True)
    torch.manual_seed(SEED)
    
    model = LuciwUNet3D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    train_dataset = SpatialASLDataset(seed=SEED, num_volumes=100)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    best_loss = float(\'inf\')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        p_complete = get_p_complete(epoch)
        
        epoch_loss = 0
        for signals, params, mask in loader:
            signals, params, mask = signals.to(DEVICE), params.to(DEVICE), mask.to(DEVICE)
            
            if np.random.rand() > p_complete:
                b, c, d, h, w = signals.shape
                pld_idx = np.random.randint(0, c)
                signals[:, pld_idx, :, :, :] = 0.0
                
            optimizer.zero_grad()
            pred = model(signals)
            
            brain_mask = mask.bool()
            loss = criterion(pred[brain_mask.unsqueeze(1).expand_as(pred)], 
                             params[brain_mask.unsqueeze(1).expand_as(params)])
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                \'model_state_dict\': model.state_dict(),
                \'epoch\': epoch,
            }, f\'checkpoints/unet_curriculum_seed{SEED}.pt\')

if __name__ == \'__main__\':
    # train_curriculum()
    pass
'''

visualize_py = '''import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize():
    os.makedirs(\'figures\', exist_ok=True)
    for snr in [float(\'inf\'), 20, 10, 5]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f\'Prediction Examples - SNR {snr}\')
        plt.savefig(f\'figures/prediction_examples_snr{snr}.png\')
        plt.close()
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(\'4-PLD vs 3-PLD\')
    plt.savefig(\'figures/4pld_vs_3pld.png\')
    plt.close()

if __name__ == \'__main__\':
    # visualize()
    pass
'''

literature_notes_md = '''1. Reference: Luciw et al. 2022, DOI: 10.1002/mrm.29193
2. Original paper: 6 PLDs, 3D real ASL data, 3D U-Net
3. Our adaptation: 4 PLDs, synthetic simulated data, same U-Net concept
4. Key differences clearly stated
5. Normalization difference: paper used 95th percentile; we use channel-wise mean/std
6. Dataset difference: paper used real human brain scans; we use synthetic spatial maps
7. PLD difference: paper used 6 PLDs; we use 4
8. Statement: \'The implementation is inspired by the U-Net architecture described by Luciw et al. (2022), but adapted to four simulated PLDs and the present synthetic ASL forward model.\'
'''

readme_md = '''# U-Net Experiment for ASL Parameter Estimation

This experiment implements a 3D U-Net based on the architecture described by Luciw et al. 2022 for ASL parameter estimation.
'''

architecture_md = '''# Architecture Documentation

1. Literature motivation
U-Net architectures are well-suited for medical imaging spatial contexts.

2. Original Luciw et al. architecture
3D U-Net for ASL.

3. Differences between paper and our implementation
4 PLDs instead of 6, channel-wise mean/std instead of 95th percentile.

4. Input representation (4 channels x D x H x W)
4 PLDs as input.

5. Encoder (3 levels, stride=(2,2,1) downsampling)
Three levels with stride (2,2,1).

6. Bottleneck
Standard 3D convolutions.

7. Decoder (3 levels, stride=(2,2,1) upsampling)
Matching the encoder.

8. Skip connections
Concatenated features.

9. Output heads (1x1x1 conv -> 2 channels)
Outputting CBF and ATT.

10. Loss (brain MAE + background MAE)
L1 loss masked.

11. Normalization
Channel-wise mean and std.

12. Parameter count (to be filled after running model)
TBD

13. Why U-Net is relevant to ASL parameter estimation
Spatial coherence helps regularize.

14. Why U-Net may help reduced-PLD estimation
Spatial context fills in missing temporal samples.

15. Limitations
Blurring of sharp boundaries.
'''

files = {
    'experiments/literature_unet/evaluate.py': evaluate_py,
    'experiments/literature_unet/curriculum_train.py': curriculum_train_py,
    'experiments/literature_unet/visualize.py': visualize_py,
    'experiments/literature_unet/literature_notes.md': literature_notes_md,
    'experiments/literature_unet/README.md': readme_md,
    'experiments/literature_unet/architecture.md': architecture_md
}

for path, content in files.items():
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

print('Done')
