import numpy as np

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
