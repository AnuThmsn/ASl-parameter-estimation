import numpy as np

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    bias = np.mean(error)
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    ccc = (2.0 * cov / denom) if denom > 0 else np.nan
    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "Bias": bias, "R2": r2, "CCC": ccc, "StdError": np.std(error)}
