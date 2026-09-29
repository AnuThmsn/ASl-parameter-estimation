import os

files = {
    "src/physics/parameters.py": """import numpy as np
PLDs = np.array([1.525, 2.025, 2.525, 3.025])
tau = 1.8
T1t = 1.2
T1a = 1.66
alpha = 0.85
beta = 0.75
lmbda = 0.9
SCALE = 100_000.0

CBF_MIN, CBF_MAX = 0.0, 100.0
ATT_MIN, ATT_MAX = 0.5, 3.0
""",
    "src/physics/asl_forward.py": """import numpy as np
from src.physics.parameters import *

def asl_signal(pld, ld, f_per_s, delta):
    prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att = np.exp(-delta / T1a)
    t1 = np.exp(-max(pld - delta, 0.0) / T1t)
    t2 = np.exp(-max(ld + pld - delta, 0.0) / T1t)
    return prefix * e_att * (t1 - t2)

def compute_signals(plds, ld, f_per_s, delta):
    return np.array([asl_signal(w, ld, f_per_s, delta) for w in plds])

def compute_signals_vec(cbf, att, ld=tau):
    f_per_s = (cbf / (6000.0 * lmbda))[:, None]
    delta = att[:, None]
    pld = PLDs[None, :]
    prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att = np.exp(-delta / T1a)
    t1 = np.exp(-np.maximum(pld - delta, 0.0) / T1t)
    t2 = np.exp(-np.maximum(ld + pld - delta, 0.0) / T1t)
    return prefix * e_att * (t1 - t2)
""",
    "src/physics/noise_models.py": """import numpy as np

def add_noise(sig, noise_sd, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    e1 = rng.normal(0.0, noise_sd, size=sig.shape)
    e2 = rng.normal(0.0, noise_sd, size=sig.shape)
    e3 = rng.normal(0.0, noise_sd, size=sig.shape)
    e4 = rng.normal(0.0, noise_sd, size=sig.shape)

    mc = np.sqrt((sig + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((sig + e3) ** 2 + e4 ** 2)
    
    return (mc + ml).astype(np.float32)

def calculate_noise_sd_from_snr(snr, ref_signal_scale=334.2039):
    if np.isinf(snr) or snr == 0:
        return 0.0
    return ref_signal_scale / snr
""",
    "architectures/00_current_baseline/architecture.md": """# Current Baseline Architecture

## Model Description
The current DNN is two independent standardized MLPs predicting CBF and ATT:
- **CBFnet:** 9 hidden layers x 50 neurons, ELU activations.
- **ATTnet:** 9 hidden layers x 100 neurons, ELU activations.
- **Loss:** MAE (L1Loss)

The forward model uses 4 PLDs: `[1.525, 2.025, 2.525, 3.025]`.
Noise model is dual-Rician magnitude: `X = sqrt((S+e1)^2+e2^2) + sqrt((S+e3)^2+e4^2)`.
""",
    "architectures/00_current_baseline/changes.md": """# Experiment Log: 00_current_baseline

**Change:** Frozen baseline reconstructed from existing notebooks.
**Reason:** Reproducible starting point.
"""
}

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("Restored physics and baseline files.")
