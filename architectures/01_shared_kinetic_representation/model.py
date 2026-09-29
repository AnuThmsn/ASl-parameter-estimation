import torch
import torch.nn as nn

class SharedKineticNet(nn.Module):
    def __init__(self, input_dim=4, shared_layers=6, shared_neurons=100,
                 cbf_layers=3, cbf_neurons=50, att_layers=3, att_neurons=100,
                 y_mean=None, y_std=None, y_min=None, y_max=None):
        super().__init__()
        
        # Output un-standardization parameters
        if y_mean is not None:
            self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer('y_std', torch.tensor(y_std, dtype=torch.float32))
        self.y_min = y_min
        self.y_max = y_max
        
        # Shared Encoder
        enc = [nn.Linear(input_dim, shared_neurons), nn.ELU()]
        for _ in range(shared_layers - 1):
            enc += [nn.Linear(shared_neurons, shared_neurons), nn.ELU()]
        self.encoder = nn.Sequential(*enc)
        
        # CBF Head
        cbf_h = [nn.Linear(shared_neurons, cbf_neurons), nn.ELU()]
        for _ in range(cbf_layers - 1):
            cbf_h += [nn.Linear(cbf_neurons, cbf_neurons), nn.ELU()]
        cbf_h.append(nn.Linear(cbf_neurons, 1))
        self.cbf_head = nn.Sequential(*cbf_h)
        
        # ATT Head
        att_h = [nn.Linear(shared_neurons, att_neurons), nn.ELU()]
        for _ in range(att_layers - 1):
            att_h += [nn.Linear(att_neurons, att_neurons), nn.ELU()]
        att_h.append(nn.Linear(att_neurons, 1))
        self.att_head = nn.Sequential(*att_h)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        latent = self.encoder(x)
        cbf_std = self.cbf_head(latent).squeeze(-1)
        att_std = self.att_head(latent).squeeze(-1)
        return torch.stack([cbf_std, att_std], dim=-1), latent

    def predict_physical(self, x):
        with torch.no_grad():
            std_pred, _ = self.forward(x)
        phys_pred = std_pred * self.y_std + self.y_mean
        
        if self.y_min is not None and self.y_max is not None:
            phys_pred[:, 0] = phys_pred[:, 0].clamp(self.y_min[0], self.y_max[0])
            phys_pred[:, 1] = phys_pred[:, 1].clamp(self.y_min[1], self.y_max[1])
            
        return phys_pred
