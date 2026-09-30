import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from experiments.literature_unet.model import LuciwUNet3D
from experiments.literature_unet.dataset import SpatialASLDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 150
SEED = 42

def get_p_complete(epoch):
    if epoch < 40: return 1.0
    elif epoch < 80: return 0.75
    elif epoch < 120: return 0.50
    else: return 0.25

def train_curriculum():
    os.makedirs('checkpoints', exist_ok=True)
    torch.manual_seed(SEED)
    
    model = LuciwUNet3D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    train_dataset = SpatialASLDataset(seed=SEED, num_volumes=100)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    best_loss = float('inf')
    
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
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
            }, f'checkpoints/unet_curriculum_seed{SEED}.pt')

if __name__ == '__main__':
    # train_curriculum()
    pass
