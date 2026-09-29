import numpy as np

def fit_normalization(X_tr):
    X_mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
    X_std = X_tr.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
    return X_mean, X_std

def apply_normalization(X, X_mean, X_std):
    return (X - X_mean) / X_std

def fit_target_standardization(Y_tr):
    Y_mean = Y_tr.mean(axis=0).astype(np.float32)
    Y_std = Y_tr.std(axis=0).astype(np.float32) + 1e-8
    return Y_mean, Y_std

def apply_target_standardization(Y, Y_mean, Y_std):
    return (Y - Y_mean) / Y_std
