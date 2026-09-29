import torch
import torch.nn as nn

def _make_mlp(in_dim, hidden_dim, n_hidden, out_dim, activation=nn.ELU):
    if n_hidden == 0: return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers = [nn.Linear(in_dim, hidden_dim), activation()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

class HierarchicalKineticNet(nn.Module):
    def __init__(self, input_dim=4, shared_hidden=90, shared_layers=7, kin_hidden=70, kin_layers=2, att_hidden=40, att_layers=2, cbf_hidden=90, cbf_layers=3, y_mean=None, y_std=None, y_min=None, y_max=None):
        super().__init__()
        if y_mean is not None:
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))
        self.y_min = y_min; self.y_max = y_max

        self.shared_encoder = _make_mlp(input_dim, shared_hidden, shared_layers - 1, shared_hidden)
        self.kinetic_encoder = _make_mlp(shared_hidden, kin_hidden, kin_layers - 1, kin_hidden)
        self.att_head = _make_mlp(kin_hidden, att_hidden, att_layers - 1, 1)

        cbf_in = shared_hidden + kin_hidden
        self.cbf_encoder = _make_mlp(cbf_in, cbf_hidden, cbf_layers - 1, cbf_hidden)
        self.cbf_head = nn.Linear(cbf_hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared_encoder(x)
        z_kin = self.kinetic_encoder(h)
        att_std = self.att_head(z_kin).squeeze(-1)
        cbf_in = torch.cat([h, z_kin], dim=-1)
        z_cbf = self.cbf_encoder(cbf_in)
        cbf_std = self.cbf_head(z_cbf).squeeze(-1)
        pred = torch.stack([cbf_std, att_std], dim=-1)
        return pred, z_kin

    def predict_physical(self, x):
        with torch.no_grad(): pred, _ = self.forward(x)
        phys = pred * self.y_std + self.y_mean
        if self.y_min is not None:
            phys[:, 0] = phys[:, 0].clamp(self.y_min[0], self.y_max[0])
            phys[:, 1] = phys[:, 1].clamp(self.y_min[1], self.y_max[1])
        return phys

class HierarchicalKineticNet_NoCondition(nn.Module):
    def __init__(self, input_dim=4, shared_hidden=90, shared_layers=7, kin_hidden=70, kin_layers=2, att_hidden=40, att_layers=2, cbf_hidden=90, cbf_layers=3, y_mean=None, y_std=None, y_min=None, y_max=None):
        super().__init__()
        if y_mean is not None:
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))
        self.y_min = y_min; self.y_max = y_max

        self.shared_encoder  = _make_mlp(input_dim, shared_hidden, shared_layers - 1, shared_hidden)
        self.kinetic_encoder = _make_mlp(shared_hidden, kin_hidden, kin_layers - 1, kin_hidden)
        self.att_head        = _make_mlp(kin_hidden, att_hidden, att_layers - 1, 1)

        cbf_hidden_nc = int(cbf_hidden + kin_hidden * 0.5)
        self.cbf_encoder_nc = _make_mlp(shared_hidden, cbf_hidden_nc, cbf_layers - 1, cbf_hidden_nc)
        self.cbf_head_nc    = nn.Linear(cbf_hidden_nc, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared_encoder(x)
        z_kin = self.kinetic_encoder(h)
        att_std = self.att_head(z_kin).squeeze(-1)
        z_cbf = self.cbf_encoder_nc(h)
        cbf_std = self.cbf_head_nc(z_cbf).squeeze(-1)
        pred = torch.stack([cbf_std, att_std], dim=-1)
        return pred, z_kin

    def predict_physical(self, x):
        with torch.no_grad(): pred, _ = self.forward(x)
        phys = pred * self.y_std + self.y_mean
        if self.y_min is not None:
            phys[:, 0] = phys[:, 0].clamp(self.y_min[0], self.y_max[0])
            phys[:, 1] = phys[:, 1].clamp(self.y_min[1], self.y_max[1])
        return phys
