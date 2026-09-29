import numpy as np
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
