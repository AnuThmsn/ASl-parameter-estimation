import torch
import torch.nn as nn

class StandardizedNet(nn.Module):
    """Baseline independent MLP"""
    def __init__(self, input_dim, n_hidden_layers, n_neurons, y_mean, y_std, y_min, y_max):
        super().__init__()
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer('y_std', torch.tensor(y_std, dtype=torch.float32))
        self.y_min = y_min
        self.y_max = y_max
        
        layers = [nn.Linear(input_dim, n_neurons), nn.ELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ELU()]
        layers.append(nn.Linear(n_neurons, 1))
        self.backbone = nn.Sequential(*layers)
        
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.backbone(x)

    def predict_physical(self, x):
        with torch.no_grad():
            std_pred = self.forward(x).squeeze(-1)
        phys_pred = std_pred * self.y_std + self.y_mean
        return phys_pred.clamp(self.y_min, self.y_max)
