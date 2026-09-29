import numpy as np

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
