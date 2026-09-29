import numpy as np
from src.physics.asl_forward import compute_signals_vec, SCALE, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.physics.noise_models import add_noise, calculate_noise_sd_from_snr

def generate_data(N_total, rng, n_noise_levels=100, sd_max=334.2039/5.0):
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)
    
    if n_noise_levels > 0:
        sd_levels = np.linspace(0.0, sd_max, n_noise_levels)
        sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
        sd_arr = sd_levels[sd_choice_idx]
    else:
        sd_arr = np.zeros(N_total)

    sig = compute_signals_vec(cbf_gt, att_gt) * SCALE
    X = add_noise(sig, sd_arr[:, None], rng=rng)
    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    
    return X, Y, sd_arr
