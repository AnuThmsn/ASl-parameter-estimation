#!/usr/bin/env python
# coding: utf-8

# # ASL CBF/ATT Estimation — DNN-G (paper-matched, ELM removed)
# 
# **Scientific objective.** Get a single supervised DNN (no ELM, no hybrid) to
# legitimately beat the Bayesian and NLS reference methods on CBF/ATT estimation
# from noisy multi-PLD ASL signals, matching the recipe validated in Ishida et al.
# 2024 (NMR Biomed).
# 
# **What changed from the earlier Experiment-H notebook:**
# - All ELM / DNN+ELM ensemble / residual-ELM / DNN-feature-ELM code has been
#   removed. This notebook trains and evaluates the DNN alone.
# - Architecture: CBFnet 5x128 -> **9x50**, ATTnet 4x64 -> **9x100** (Ishida et al.
#   2024, Table 1 — their empirically grid-searched optimum for this exact
#   estimation problem).
# - Loss: MSE -> **MAE** (`nn.L1Loss()`) for both nets, also matching Table 1.
# - New save paths, so old 5x128/4x64-MSE weights are never silently reloaded —
#   this forces a genuine retrain under the corrected configuration.
# 
# **Why the fix was needed.** The previous DNN-G was losing to Bayesian badly on
# ATT (RMSE ~0.30 vs ~0.08 at SNR=10) and losing to plain NLS on CBF (RMSE ~4.27
# vs ~2.98 at SNR=10) — the opposite of what the paper reports for supervised DNN
# vs. conventional fitting. Both deviations above are architecture/training
# choices that diverged from the paper's own validated recipe for the same
# problem; Section 8f at the end checks whether closing them closes the gap.
# 
# 
# ---
# 
# > **Round 2 update.** The architecture+MAE fix above only moved CBF RMSE 4.2676 -> 4.2256 and ATT CCC 0.9046 -> 0.9071 at SNR=10 — real but marginal, and ATT RMSE barely moved (0.3026 -> 0.3029) despite the ATT net going from 4x64 to 9x100. That capacity increase doing essentially nothing for ATT is the tell that architecture wasn't the dominant bottleneck. Two changes in this round:
# >
# > 1. **Training-data noise density** (Section 3): 12 discrete SNR bins -> 100 evenly spaced noise-SD levels (51 for test), matching the paper's dense sweep. Train set also bumped 1M -> 2M samples so per-level density doesn't collapse now that there are ~8x more levels.
# > 2. **New diagnostic** (Section 7): prints the actual noise-SD-to-signal ratio at each evaluation SNR checkpoint. Run this and compare it against whatever noise-generation code produced the Bayesian/NLS reference numbers. **This is the most likely explanation left for the persistent gap** — if "SNR=10" means a different absolute noise level in the two notebooks, no amount of DNN tuning here will ever close it, because the two methods aren't being tested on the same noise level despite the matching label.

# ## SECTION 1 — Environment and reproducibility

# In[ ]:


import os, copy, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import least_squares
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# QUICK_TEST=True runs the whole pipeline end-to-end on tiny sample
# sizes / small seed counts / coarse grids so you can smoke-test the
# notebook logic in a couple of minutes on CPU.
#
# Set QUICK_TEST=False (and run on a GPU, e.g. Kaggle) to train at full
# scale (N_TRAIN=1,000,000 / N_VAL=100,000).
# ------------------------------------------------------------------
QUICK_TEST = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"QUICK_TEST mode: {QUICK_TEST}")

OUTPUT_DIR = "/kaggle/working/asl_dnn_papermatched" if os.path.isdir("/kaggle/working") else "./asl_dnn_papermatched"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")


# ## SECTION 2 — Physics simulator (validated, unchanged)
# 
# Reused verbatim from the Experiment F/G notebook. Constants, forward model, PLDs,
# `SCALE`, and noise mechanism are not modified. The zero-noise training input is
# `2 x physical_signal` (sum of two Rician-magnitude repeats, `mc + ml`), exactly as
# validated previously.

# In[ ]:


# ---- Physics constants — UNCHANGED ----
PLDs  = np.array([1.525, 2.025, 2.525, 3.025])   # seconds
tau   = 1.8
T1t   = 1.2
T1a   = 1.66
alpha = 0.85
beta  = 0.75
lmbda = 0.9
SCALE = 100_000.0

CBF_MIN, CBF_MAX = 0.0,  100.0
ATT_MIN, ATT_MAX = 0.5,    3.0

REF_CBF, REF_ATT, REF_PLD, REF_LD = 50.0, 1.6, 2.0, 1.8   # UNCHANGED reference case

def asl_signal(pld, ld, f_per_s, delta):
    prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att  = np.exp(-delta / T1a)
    t1     = np.exp(-max(pld - delta,       0.0) / T1t)
    t2     = np.exp(-max(ld  + pld - delta, 0.0) / T1t)
    return prefix * e_att * (t1 - t2)

def compute_signals(plds, ld, f_per_s, delta):
    return np.array([asl_signal(w, ld, f_per_s, delta) for w in plds])

def compute_signals_vec(cbf, att, ld=tau):
    """Vectorized version of the SAME formula, for fast batch generation.
    cbf, att: shape (N,). Returns shape (N, len(PLDs))."""
    f_per_s = (cbf / (6000.0 * lmbda))[:, None]
    delta   = att[:, None]
    pld     = PLDs[None, :]
    prefix  = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att   = np.exp(-delta / T1a)
    t1      = np.exp(-np.maximum(pld - delta,      0.0) / T1t)
    t2      = np.exp(-np.maximum(ld  + pld - delta, 0.0) / T1t)
    return prefix * e_att * (t1 - t2)

_f_ref     = REF_CBF / (6000.0 * lmbda)
SIG_REF_SC = abs(asl_signal(REF_PLD, REF_LD, _f_ref, REF_ATT)) * SCALE
print(f"Reference signal (scaled): {SIG_REF_SC:.4f}   (validated value: ~334.2039)")

# Sanity check against the validated reference case
_chk_f = 50.0 / (6000.0 * lmbda)
_chk = compute_signals(PLDs, tau, _chk_f, 1.5) * SCALE
print("Sanity check CBF=50, ATT=1.5 physical signal:", np.round(_chk, 5))
print("  (validated: [485.16653 319.84150 210.85252 139.00255])")
print("Sanity check zero-noise training input (2x):  ", np.round(2.0 * _chk, 5))
print("  (validated: [970.33307 639.68300 421.70505 278.00510])")

assert np.allclose(_chk, [485.16653, 319.8415, 210.85252, 139.00255], atol=1e-2), \
    "Physics simulator does not match validated reference signal!"
print("\n[OK] Physics simulator matches validated reference values.")


# ## SECTION 3 — Training-data generation
# 
# Noise SD is now drawn from 100 evenly spaced levels (51 for the test split) between zero and a fixed worst-case level, matching Ishida et al. 2024's dense noise sweep — previously this drew from only 12 discrete target-SNR bins. Rician `mc + ml` noise mechanism is unchanged. Held-out test split unchanged (separate from train/val).

# In[ ]:


# ------------------------------------------------------------------
# Noise-level sampling for TRAINING/VAL/TEST DATA GENERATION.
#
# Previously: 12 discrete "target SNR" bins drawn with equal probability
# (~8.3% each). Ishida et al. 2024 instead sweep noise SD across 100 evenly
# spaced levels (train + val) / 51 levels (test), covering the FULL range
# continuously rather than a handful of discrete checkpoints. This matters
# because with only 12 bins, ~1,000,000/12 ~= 83k samples land on each exact
# noise level and nothing in between is seen during training — the network
# never learns intermediate noise levels directly.
#
# Fix: sample noise SD from N_LEVELS evenly spaced levels between 0 (no
# noise) and SD_MAX (set to the same worst-case level the old SNR=5 bin
# implied, so the hardest noise case is unchanged — only the *density* of
# coverage between clean and worst-case changes).
#
# NOTE: SNR_VALUES/SNR_PROBS are kept below only for the SNR-level
# *evaluation* checkpoints (Section 7) — that part is unrelated to how
# training data is generated and is untouched.
# ------------------------------------------------------------------
SNR_VALUES = [5, 8, 10, 12, 15, 20, 30, 50, 100, 200, 500, np.inf]
SNR_PROBS  = [1.0 / len(SNR_VALUES)] * len(SNR_VALUES)

assert abs(sum(SNR_PROBS) - 1.0) < 1e-9
assert len(SNR_VALUES) == len(SNR_PROBS)

SD_MAX = SIG_REF_SC / 5.0   # same worst-case noise level as the old SNR=5 bin
N_NOISE_LEVELS_TRAIN = 100  # matches Ishida et al. 2024 (100 levels for train)
N_NOISE_LEVELS_VAL   = 100  # matches Ishida et al. 2024 (100 levels for val)
N_NOISE_LEVELS_TEST  = 51   # matches Ishida et al. 2024 (51 levels for test)

def generate_data(N_total, rng, n_noise_levels=N_NOISE_LEVELS_TRAIN):
    """CBF/ATT ground truth uniform as before. Noise SD is now drawn from
    n_noise_levels evenly spaced levels in [0, SD_MAX] (paper-matched dense
    sweep) instead of a small set of discrete target-SNR bins. Rician mc/ml
    noise mechanism and X = mc + ml are unchanged."""
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)

    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]

    # effective SNR for bookkeeping only (inf where sd==0)
    with np.errstate(divide='ignore'):
        snr_arr = np.where(sd_arr > 0, SIG_REF_SC / sd_arr, np.inf)

    sig = compute_signals_vec(cbf_gt, att_gt) * SCALE

    sd = sd_arr[:, None]
    e1 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e2 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e3 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e4 = rng.normal(0.0, 1.0, size=sig.shape) * sd

    mc = np.sqrt((sig + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((sig + e3) ** 2 + e4 ** 2)
    X = (mc + ml).astype(np.float32)

    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return X, Y, snr_arr

if QUICK_TEST:
    N_TRAIN, N_VAL, N_TEST = 20_000, 4_000, 4_000
else:
    # Paper uses 3,000,000 / 50,000 / 255,000. Test set kept smaller here
    # (100k, not 255k) since it's only used for held-out reporting, not
    # fitting anything -- doesn't need paper-scale size. Train bumped from
    # 1M -> 2M as a middle ground: closer to the paper's density per noise
    # level (2M/100 = 20k/level vs paper's 3M/100 = 30k/level, vs the old
    # 1M/12 = 83k on only 12 discrete levels) without doubling Kaggle
    # runtime outright. Bump to 3_000_000 if you want an exact match and
    # have the GPU time.
    N_TRAIN, N_VAL, N_TEST = 2_000_000, 50_000, 100_000

rng_tr   = np.random.default_rng(42)
rng_val  = np.random.default_rng(99)
rng_test = np.random.default_rng(7)   # independent from train/val, held out for everything

X_tr,   Y_tr,   snr_tr   = generate_data(N_TRAIN, rng_tr,   n_noise_levels=N_NOISE_LEVELS_TRAIN)
X_val,  Y_val,  snr_val  = generate_data(N_VAL,   rng_val,  n_noise_levels=N_NOISE_LEVELS_VAL)
X_test, Y_test, snr_test = generate_data(N_TEST,  rng_test, n_noise_levels=N_NOISE_LEVELS_TEST)

print(f"Train {X_tr.shape} | Val {X_val.shape} | Test {X_test.shape}")
print(f"Noise levels: train={N_NOISE_LEVELS_TRAIN}, val={N_NOISE_LEVELS_VAL}, test={N_NOISE_LEVELS_TEST} (evenly spaced SD in [0, {SD_MAX:.4f}])")
print("Test set is used ONLY for final reporting: never for normalization fitting,")
print("DNN training or hyperparameter selection.")


# In[ ]:


# ---- Input normalization: fit on TRAIN only ----
X_mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
X_std  = X_tr.std(axis=0,  keepdims=True).astype(np.float32) + 1e-8

X_tr_n   = (X_tr   - X_mean) / X_std
X_val_n  = (X_val  - X_mean) / X_std
X_test_n = (X_test - X_mean) / X_std

print("X_mean:", X_mean.squeeze())
print("X_std :", X_std.squeeze())
assert np.all(X_std > 0), "X_std must be > 0 for every feature"
print("[OK] X_std > 0 for every feature.")

# ---- Target standardization: fit on TRAIN only ----
CBF_mean, CBF_std = float(Y_tr[:, 0].mean()), float(Y_tr[:, 0].std() + 1e-8)
ATT_mean, ATT_std = float(Y_tr[:, 1].mean()), float(Y_tr[:, 1].std() + 1e-8)

print(f"\nCBF target: mean={CBF_mean:.3f} std={CBF_std:.3f}  (physical range [{CBF_MIN},{CBF_MAX}])")
print(f"ATT target: mean={ATT_mean:.3f} std={ATT_std:.3f}  (physical range [{ATT_MIN},{ATT_MAX}])")

np.save(os.path.join(OUTPUT_DIR, "X_mean.npy"), X_mean)
np.save(os.path.join(OUTPUT_DIR, "X_std.npy"),  X_std)
np.save(os.path.join(OUTPUT_DIR, "CBF_y_mean_std.npy"), np.array([CBF_mean, CBF_std], dtype=np.float32))
np.save(os.path.join(OUTPUT_DIR, "ATT_y_mean_std.npy"), np.array([ATT_mean, ATT_std], dtype=np.float32))

# ---- Clean (noise-free) evaluation population — same grid as Experiment G ----
CBF_VALUES_CLEAN = np.linspace(10.0, 100.0, 46)
ATT_VALUES_CLEAN = np.linspace(0.5, 3.0, 52)
cbf_grid, att_grid = np.meshgrid(CBF_VALUES_CLEAN, ATT_VALUES_CLEAN, indexing="ij")
clean_cbf_true = cbf_grid.reshape(-1)
clean_att_true = att_grid.reshape(-1)

clean_sig = compute_signals_vec(clean_cbf_true, clean_att_true) * SCALE
clean_input_phys = 2.0 * clean_sig
clean_input_norm = (clean_input_phys - X_mean) / X_std
print(f"\nClean population grid: {len(clean_cbf_true)} points "
      f"({len(CBF_VALUES_CLEAN)} CBF x {len(ATT_VALUES_CLEAN)} ATT)")


# ## SECTION 4 — DNN architecture: paper-matched configuration (fixes Experiment G)
# 
# **Why this changed.** The Experiment-G baseline (CBF: 5 hidden layers x 128 neurons, ATT: 4 hidden layers x 64 neurons, MSE loss) was underperforming both the Bayesian and NLS reference methods, especially on ATT (RMSE ~0.30 vs Bayesian's ~0.08 at SNR=10) and, unexpectedly, losing to plain NLS on CBF too (RMSE ~4.27 vs ~2.98 at SNR=10) — the opposite of what Ishida et al. 2024 report for their supervised DNN vs. their conventional (WD+LS) method.
# 
# Two concrete deviations from Ishida et al.'s own grid-search-validated hyperparameters (their Table 1) stood out:
# 
# 1. **Loss function.** Their 96-configuration grid search found **MAE** beat MSE for both supervised CBF and supervised ATT networks. This notebook was using `nn.MSELoss()`. MSE over-weights the large-error tail that Rician noise produces, which is exactly the kind of bias a Bayesian method with informative priors (or a direct least-squares fit) doesn't suffer from — a plausible explanation for the CBF gap vs. NLS.
# 2. **Capacity, especially for ATT.** Their optimal supervised architecture was **9 hidden layers** for both networks — 50 neurons/layer for CBF, 100 neurons/layer for ATT. This notebook's ATT net (4 layers x 64 neurons) had roughly a third of the depth and two-thirds of the width of the paper's ATT net. ATT is the harder target (it defines the *shape*/timing of the kinetic curve, not just a scale factor), so under-capacity there is the more likely driver of the large ATT gap vs. Bayesian.
# 
# **Fix applied below:** CBFnet -> 9x50, ATTnet -> 9x100 (matching Table 1 of Ishida et al. 2024 exactly), loss -> MAE (`nn.L1Loss()`) for both. Old cached weights are not reused (new save paths below), so this forces a genuine retrain under the new configuration. Patience is also raised slightly (10 -> 15) since deeper nets take a few more epochs to shake out noise in the validation curve before it's safe to call convergence.

# In[ ]:


class StandardizedNet(nn.Module):
    """Unchanged from Experiment F/G. ELU MLP backbone, kaiming-normal init,
    predicts a standardized target and rescales+clamps to physical units."""
    def __init__(self, input_dim, n_hidden_layers, n_neurons, y_mean, y_std, y_min, y_max):
        super().__init__()
        self.y_mean = y_mean
        self.y_std  = y_std
        self.y_min  = y_min
        self.y_max  = y_max
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

    def penultimate_features(self, x):
        """Penultimate-layer activation hook (unused now that ELM is removed;
        kept in case a future experiment wants it, harmless otherwise)."""
        with torch.no_grad():
            h = x
            for layer in list(self.backbone.children())[:-1]:   # everything except final Linear
                h = layer(h)
        return h


INPUT_DIM = len(PLDs)

# Paper-matched CBF network: 9 hidden layers x 50 neurons (Ishida et al. 2024, Table 1)
CBFnet_G = StandardizedNet(INPUT_DIM, n_hidden_layers=9, n_neurons=50,
                            y_mean=CBF_mean, y_std=CBF_std, y_min=CBF_MIN, y_max=CBF_MAX).to(device)

# Paper-matched ATT network: 9 hidden layers x 100 neurons (Ishida et al. 2024, Table 1)
ATTnet_G = StandardizedNet(INPUT_DIM, n_hidden_layers=9, n_neurons=100,
                            y_mean=ATT_mean, y_std=ATT_std, y_min=ATT_MIN, y_max=ATT_MAX).to(device)

n_params_cbf = sum(p.numel() for p in CBFnet_G.parameters())
n_params_att = sum(p.numel() for p in ATTnet_G.parameters())
print(f"CBFnet_G (paper-matched, 9x50)  params: {n_params_cbf}")
print(f"ATTnet_G (paper-matched, 9x100) params: {n_params_att}")


# ## SECTION 5 — DNN training / loading
# 
# The Experiment-G weights were **not saved** in the source notebook, so this
# section retrains Experiment G from scratch using the exact same training loop,
# optimizer, and early-stopping rule as before (Adam, lr=1e-3, batch size 512,
# patience 10, MSE loss on the standardized target, gradient clipping at 1.0). If
# `CBF_G_best.pth` / `ATT_G_best.pth` already exist on disk (e.g. from a previous
# run of this notebook), they are loaded instead of retraining.

# In[ ]:


def train_standardized(net, X_tr_n, Y_col_tr, X_val_n, Y_col_val,
                        batch_size=512, max_epochs=200, patience=10, lr=1e-3, verbose=True):
    Y_tr_std  = (Y_col_tr - net.y_mean) / net.y_std
    Y_val_std = (Y_col_val - net.y_mean) / net.y_std

    ds_tr  = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32),
                            torch.tensor(Y_tr_std, dtype=torch.float32).unsqueeze(1))
    ds_val = TensorDataset(torch.tensor(X_val_n, dtype=torch.float32),
                            torch.tensor(Y_val_std, dtype=torch.float32).unsqueeze(1))

    ld_tr  = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    ld_val = DataLoader(ds_val, batch_size=batch_size)

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    crit  = nn.L1Loss()   # MAE — matches Ishida et al. 2024 Table 1 for supervised nets

    best_val   = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    no_imp = 0

    for epoch in range(max_epochs):
        net.train()
        tr = 0.0
        for xb, yb in ld_tr:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = net(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
            tr += loss.item()
        tr /= len(ld_tr)

        net.eval()
        with torch.no_grad():
            vl = 0.0
            for xb, yb in ld_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = net(xb)
                vl += crit(pred, yb).item()
            vl /= len(ld_val)

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"  Epoch {epoch+1:3d}: train={tr:.6f} val={vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(net.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    net.load_state_dict(best_state)
    return net

# New paths: paper-matched config (9x50 CBF / 9x100 ATT, MAE loss) — must NOT collide with
# old 5x128/4x64-MSE weights, since those have incompatible shapes and different training.
CBF_G_PATH = os.path.join(OUTPUT_DIR, "CBF_papermatched_9x50_MAE_best.pth")
ATT_G_PATH = os.path.join(OUTPUT_DIR, "ATT_papermatched_9x100_MAE_best.pth")
MAX_EPOCHS = 20 if QUICK_TEST else 200

if os.path.exists(CBF_G_PATH) and os.path.exists(ATT_G_PATH):
    print("Loading existing Experiment-G weights ...")
    CBFnet_G.load_state_dict(torch.load(CBF_G_PATH, map_location=device))
    ATTnet_G.load_state_dict(torch.load(ATT_G_PATH, map_location=device))
else:
    print("No saved paper-matched weights found — training from scratch.\n")
    print("Training ATTnet (paper-matched: 9 layers x 100 neurons, MAE loss)...")
    ATTnet_G = train_standardized(ATTnet_G, X_tr_n, Y_tr[:, 1], X_val_n, Y_val[:, 1],
                                   lr=1e-3, max_epochs=MAX_EPOCHS, patience=15)
    print("\nTraining CBFnet_G (paper-matched: 9 layers x 50 neurons, MAE loss)...")
    CBFnet_G = train_standardized(CBFnet_G, X_tr_n, Y_tr[:, 0], X_val_n, Y_val[:, 0],
                                   lr=1e-3, max_epochs=MAX_EPOCHS, patience=15)
    torch.save(CBFnet_G.state_dict(), CBF_G_PATH)
    torch.save(ATTnet_G.state_dict(), ATT_G_PATH)
    print(f"\nSaved paper-matched weights to {CBF_G_PATH} and {ATT_G_PATH}")

CBFnet_G.eval(); ATTnet_G.eval()

print("\n" + "=" * 70)
print("PAPER-MATCHED DNN — CONFIGURATION SUMMARY")
print("=" * 70)
print(f"CBFnet architecture : 9 hidden layers x 50 neurons,  ELU, {n_params_cbf} params")
print(f"ATTnet architecture : 9 hidden layers x 100 neurons, ELU, {n_params_att} params")
print(f"Training samples    : N_train={N_TRAIN}, N_val={N_VAL}, N_test={N_TEST}")
print(f"Optimizer           : Adam, lr=1e-3, batch_size=512, grad_clip=1.0")
print(f"Loss function       : MAE (L1Loss) — matches Ishida et al. 2024 Table 1")
print(f"Early stopping      : patience=15 epochs on val MAE (standardized target)")
print(f"X_mean              : {X_mean.squeeze()}")
print(f"X_std               : {X_std.squeeze()}")
print(f"CBF target mean/std : {CBF_mean:.4f} / {CBF_std:.4f}")
print(f"ATT target mean/std : {ATT_mean:.4f} / {ATT_std:.4f}")


# ## Metric helpers (used throughout)

# In[ ]:


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mse  = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(error))
    bias = np.mean(error)
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t,  var_p  = np.var(y_true),  np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    ccc = (2.0 * cov / denom) if denom > 0 else np.nan
    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "Bias": bias, "R2": r2, "CCC": ccc}

def dnn_predict_physical(net, X_norm):
    x_t = torch.tensor(X_norm.astype(np.float32), dtype=torch.float32, device=device)
    return net.predict_physical(x_t).cpu().numpy().reshape(-1)

# Experiment-G DNN predictions, reused everywhere below
CBF_DNN_tr   = dnn_predict_physical(CBFnet_G, X_tr_n)
ATT_DNN_tr   = dnn_predict_physical(ATTnet_G, X_tr_n)
CBF_DNN_val  = dnn_predict_physical(CBFnet_G, X_val_n)
ATT_DNN_val  = dnn_predict_physical(ATTnet_G, X_val_n)
CBF_DNN_test = dnn_predict_physical(CBFnet_G, X_test_n)
ATT_DNN_test = dnn_predict_physical(ATTnet_G, X_test_n)

print("DNN-G clean-data-independent sanity check (test split):")
print("  CBF:", calc_metrics(Y_test[:, 0], CBF_DNN_test))
print("  ATT:", calc_metrics(Y_test[:, 1], ATT_DNN_test))


# ## SECTION 6 — Clean population evaluation
# 
# DNN-G evaluated on a noise-free CBF x ATT grid: CBF in [10,100] x ATT in [0.5,3.0], 46x52 grid.

# In[ ]:


MODEL_NAMES = ["DNN-G"]

def predict_all_models(X_norm):
    """Given a normalized (N,4) input, return CBF/ATT predictions from DNN-G.
    Kept as a dict-of-dicts (single key) so downstream sections that loop over
    MODEL_NAMES don't need to change."""
    cbf_dnn = dnn_predict_physical(CBFnet_G, X_norm)
    att_dnn = dnn_predict_physical(ATTnet_G, X_norm)
    return {"DNN-G": {"CBF": cbf_dnn, "ATT": att_dnn}}

clean_preds = predict_all_models(clean_input_norm)

clean_results_rows = []
for name in MODEL_NAMES:
    m_cbf = calc_metrics(clean_cbf_true, clean_preds[name]["CBF"])
    m_att = calc_metrics(clean_att_true, clean_preds[name]["ATT"])
    row = {"Model": name}
    row.update({f"CBF_{k}": v for k, v in m_cbf.items()})
    row.update({f"ATT_{k}": v for k, v in m_att.items()})
    clean_results_rows.append(row)

clean_results_df = pd.DataFrame(clean_results_rows)
print("=" * 100)
print("DNN-G — CLEAN POPULATION PERFORMANCE")
print("=" * 100)
display(clean_results_df.round(5))

clean_results_df.to_csv(os.path.join(OUTPUT_DIR, "clean_results.csv"), index=False)


# ## SECTION 7 — SNR evaluation
# 
# Same validated SNR robustness protocol as before (single ground-truth point CBF=50, ATT=1.5, repeated noisy draws per SNR level), evaluated for DNN-G only.

# In[ ]:


# ============================================================
# ZERO-NOISE SANITY CHECK
# ============================================================
print("============================================================")
print("ZERO-NOISE SANITY CHECK")
print("============================================================")

# Directly calculate the 1x physical signal for the reference point CBF=50, ATT=1.5
f_s_ref = 50.0 / (6000.0 * lmbda)
sig_ref = compute_signals(PLDs, tau, f_s_ref, 1.5) * SCALE

print("Physical signal (1x):")
print(sig_ref)

# Zero-noise Rician measurements
mc = np.sqrt(sig_ref**2)
ml = np.sqrt(sig_ref**2)

zero_noise_input = mc + ml

print("\nZero-noise training input (Mc + Ml):")
print(zero_noise_input)

expected_input = np.array([970.33307, 639.683, 421.70505, 278.0051], dtype=np.float32)
print("\nExpected zero-noise training input:")
print(expected_input)

diff = np.max(np.abs(zero_noise_input - expected_input))
print(f"\nMaximum difference: {diff:.6f}")
if diff < 1e-2:
    print("-> SANITY CHECK PASSED! The Rician combination correctly reproduces the training input.")
else:
    print("-> SANITY CHECK FAILED! Do not proceed with SNR sweep.")

# ============================================================
# CORRECTED FULL-GRID SNR EVALUATION
# ============================================================

def grid_snr_all_models(snr_levels, cbf_true, att_true, noiseless_signals_phys, seed=123):
    rng = np.random.default_rng(seed)
    records = []
    
    noiseless_signals_phys = np.asarray(noiseless_signals_phys, dtype=np.float32)
    cbf_true = np.asarray(cbf_true, dtype=np.float32)
    att_true = np.asarray(att_true, dtype=np.float32)
    
    print("\n" + "=" * 75)
    print("CORRECTED FULL-GRID SNR EVALUATION")
    print("=" * 75)
    
    for snr in snr_levels:
        if np.isinf(snr):
            sd = 0.0
        else:
            sd = SIG_REF_SC / float(snr)
            
        e1 = rng.normal(0.0, sd, noiseless_signals_phys.shape)
        e2 = rng.normal(0.0, sd, noiseless_signals_phys.shape)
        e3 = rng.normal(0.0, sd, noiseless_signals_phys.shape)
        e4 = rng.normal(0.0, sd, noiseless_signals_phys.shape)
        
        mc = np.sqrt((noiseless_signals_phys + e1) ** 2 + e2 ** 2)
        ml = np.sqrt((noiseless_signals_phys + e3) ** 2 + e4 ** 2)
        
        noisy_input_phys = (mc + ml).astype(np.float32)
        noisy_norm = ((noisy_input_phys - X_mean) / X_std).astype(np.float32)
        
        preds = predict_all_models(noisy_norm)
        
        for name in MODEL_NAMES:
            cbf_pred = np.asarray(preds[name]["CBF"]).reshape(-1)
            att_pred = np.asarray(preds[name]["ATT"]).reshape(-1)
            
            m_cbf = calc_metrics(cbf_true, cbf_pred)
            m_att = calc_metrics(att_true, att_pred)
            
            records.append({
                "SNR": snr, "Model": name,
                "CBF_RMSE": m_cbf["RMSE"], "CBF_MSE": m_cbf["MSE"], 
                "CBF_MAE": m_cbf["MAE"], "CBF_CCC": m_cbf["CCC"],
                "ATT_RMSE": m_att["RMSE"], "ATT_MSE": m_att["MSE"], 
                "ATT_MAE": m_att["MAE"], "ATT_CCC": m_att["CCC"]
            })
            
    return pd.DataFrame(records)

# Use the 2,392-point clean population true values and 1x physical signal
SNR_TEST_LEVELS_FULL = SNR_VALUES

print(f"\nRunning corrected SNR sweep across the {len(clean_cbf_true)}-point physiological grid...")
snr_results_df = grid_snr_all_models(SNR_TEST_LEVELS_FULL, clean_cbf_true, clean_att_true, clean_sig)

print("\nSNR robustness — CBF RMSE by model and SNR:")
display(snr_results_df.pivot(index="SNR", columns="Model", values="CBF_RMSE").round(4))

snr_results_df.to_csv(os.path.join(OUTPUT_DIR, "snr_results.csv"), index=False)


# ### Diagnostic — is "SNR" even defined the same way as in the Bayesian/NLS notebook?
# 
# The architecture+loss fix only moved CBF RMSE from 4.2676 -> 4.2256 and ATT CCC from 0.9046 -> 0.9071 at SNR=10 — real but tiny, especially given the ATT net went from 4x64 to 9x100 (4x the depth) with no meaningful change in ATT RMSE at all. That's a strong signal that architecture/loss was never the dominant bottleneck.
# 
# One concrete hypothesis: `SNR` here is defined as `SIG_REF_SC / noise_SD`, where `SIG_REF_SC` is the signal magnitude at a single reference point (CBF=50, ATT=1.5s), applied globally to every voxel in the grid regardless of its own true CBF/ATT. If the Bayesian/NLS notebook defines "SNR" differently (e.g. per-voxel signal-to-noise using each voxel's own amplitude, or a different reference point/PLD, or power ratio instead of amplitude ratio), then "SNR=10" is not the same actual noise level in both notebooks — and no amount of DNN tuning here can close a gap that's really a labeling mismatch, not a modeling one.
# 
# Printed below: the literal noise SD and the SD-to-signal ratio at each evaluation SNR checkpoint, for the reference point AND for a low-signal case (CBF=15, ATT=2.5s, where the paper shows all methods struggle most). Compare these numbers against whatever noise-generation code produced the Bayesian/NLS numbers — if they don't use the same formula, that's the fix to make before touching this notebook again.

# In[ ]:


# Reference point (used to define SNR here): CBF=50, ATT=1.5s
print(f"Reference signal magnitude (per PLD), CBF=50 ATT=1.5s: {sig_ref}")
print(f"SIG_REF_SC (scalar used in SNR = SIG_REF_SC / noise_SD): {SIG_REF_SC:.4f}\n")

# A weak-signal case for comparison: low CBF, long ATT
f_s_weak = 15.0 / (6000.0 * lmbda)
sig_weak = compute_signals(PLDs, tau, f_s_weak, 2.5) * SCALE
print(f"Weak-signal case (CBF=15, ATT=2.5s): {sig_weak}\n")

print(f"{'SNR':<8} | {'noise SD':<12} | {'SD / ref signal[0]':<20} | {'SD / weak signal[0]':<20}")
print("-" * 70)
for snr in [5, 8, 10, 12, 15, 20, 30, 50]:
    sd = SIG_REF_SC / snr
    print(f"{snr:<8} | {sd:<12.4f} | {sd / sig_ref[0]:<20.4%} | {sd / sig_weak[0]:<20.4%}")


# ## SECTION 8 — Final DNN-G evaluation
# 
# ### 8a. CBF monotonicity (ATT fixed at 1.5s)
# 
# A DNN that isn't even monotonic in CBF (predicted CBF should strictly increase as true CBF increases, at fixed ATT) has no chance of matching a physics-based fitting method — checked here as a basic sanity gate before trusting the RMSE/CCC numbers elsewhere.

# In[ ]:


ATT_FIXED = 1.5
CBF_SWEEP = np.arange(10.0, 100.1, 5.0)

sig_sweep = compute_signals_vec(CBF_SWEEP, np.full_like(CBF_SWEEP, ATT_FIXED)) * SCALE
clean_sweep_phys = 2.0 * sig_sweep
clean_sweep_norm = (clean_sweep_phys - X_mean) / X_std

mono_preds = predict_all_models(clean_sweep_norm)

monotonicity_violations = {}
plt.figure(figsize=(9, 6))
plt.plot(CBF_SWEEP, CBF_SWEEP, "--", color="black", label="Ideal")
for name in MODEL_NAMES:
    pred = mono_preds[name]["CBF"]
    diffs = np.diff(pred)
    n_violations = int(np.sum(diffs < 0))
    monotonicity_violations[name] = n_violations
    plt.plot(CBF_SWEEP, pred, "o-", label=f"{name} ({n_violations} viol.)")

plt.xlabel("True CBF"); plt.ylabel("Predicted CBF")
plt.title(f"CBF monotonicity check (ATT fixed = {ATT_FIXED}s)")
plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "cbf_monotonicity.png"), dpi=120)
plt.show()

print("CBF monotonicity violations by model:")
for name, v in monotonicity_violations.items():
    print(f"  {name:18s}: {v} violations  {'(STRICTLY MONOTONIC)' if v == 0 else ''}")


# ### 8b. ATT response (CBF fixed at 50)

# In[ ]:


CBF_FIXED = 50.0
ATT_SWEEP = np.linspace(0.5, 3.0, 26)

sig_att_sweep = compute_signals_vec(np.full_like(ATT_SWEEP, CBF_FIXED), ATT_SWEEP) * SCALE
clean_att_sweep_phys = 2.0 * sig_att_sweep
clean_att_sweep_norm = (clean_att_sweep_phys - X_mean) / X_std

att_sweep_preds = predict_all_models(clean_att_sweep_norm)

plt.figure(figsize=(9, 6))
plt.plot(ATT_SWEEP, ATT_SWEEP, "--", color="black", label="Ideal")
att_response_rows = []
for name in MODEL_NAMES:
    pred = att_sweep_preds[name]["ATT"]
    plt.plot(ATT_SWEEP, pred, "o-", label=name)
    m = calc_metrics(ATT_SWEEP, pred)
    m["Model"] = name
    att_response_rows.append(m)

plt.axvline(1.525, linestyle=":", color="gray", label="PLD1 = 1.525s")
plt.xlabel("True ATT (s)"); plt.ylabel("Predicted ATT (s)")
plt.title(f"ATT response (CBF fixed = {CBF_FIXED})")
plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "att_response.png"), dpi=120)
plt.show()

att_response_df = pd.DataFrame(att_response_rows)[["Model", "RMSE", "MAE", "CCC"]]
print("ATT response sweep — RMSE/MAE/CCC by model:")
display(att_response_df.round(4))


# ### 8c. Error by CBF range (clean population)

# In[ ]:


CBF_RANGE_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

cbf_range_rows = []
for lo, hi in CBF_RANGE_BINS:
    mask = (clean_cbf_true >= lo) & (clean_cbf_true < hi if hi < 100 else clean_cbf_true <= hi)
    for name in MODEL_NAMES:
        pred = clean_preds[name]["CBF"][mask]
        true = clean_cbf_true[mask]
        m = calc_metrics(true, pred)
        cbf_range_rows.append({"Range": f"{lo}-{hi}", "Model": name, "N": int(mask.sum()),
                                "RMSE": m["RMSE"], "MAE": m["MAE"], "Bias": m["Bias"], "CCC": m["CCC"]})

cbf_range_df = pd.DataFrame(cbf_range_rows)
print("Error by CBF range:")
display(cbf_range_df.pivot(index="Range", columns="Model", values="RMSE").round(4))
cbf_range_df.to_csv(os.path.join(OUTPUT_DIR, "cbf_range_results.csv"), index=False)


# ### 8d. Error by ATT range (clean population)

# In[ ]:


ATT_RANGE_BINS = [(0.5, 1.0), (1.0, 1.4), (1.4, 1.525), (1.525, 1.8), (1.8, 2.2), (2.2, 2.6), (2.6, 3.0)]

att_range_rows = []
for lo, hi in ATT_RANGE_BINS:
    mask = (clean_att_true >= lo) & (clean_att_true < hi if hi < 3.0 else clean_att_true <= hi)
    for name in MODEL_NAMES:
        pred = clean_preds[name]["ATT"][mask]
        true = clean_att_true[mask]
        if mask.sum() == 0:
            continue
        m = calc_metrics(true, pred)
        att_range_rows.append({"Range": f"{lo}-{hi}", "Model": name, "N": int(mask.sum()),
                                "RMSE": m["RMSE"], "MAE": m["MAE"], "Bias": m["Bias"], "CCC": m["CCC"]})

att_range_df = pd.DataFrame(att_range_rows)
print("Error by ATT range:")
display(att_range_df.pivot(index="Range", columns="Model", values="RMSE").round(4))
att_range_df.to_csv(os.path.join(OUTPUT_DIR, "att_range_results.csv"), index=False)


# ### 8e. Required plots

# In[ ]:


# 1 & 2. Clean CBF / ATT: True vs DNN-G
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(clean_cbf_true, clean_preds["DNN-G"]["CBF"], s=3, alpha=0.3, label="DNN-G", color="tab:blue")
axes[1].scatter(clean_att_true, clean_preds["DNN-G"]["ATT"], s=3, alpha=0.3, label="DNN-G", color="tab:blue")
axes[0].plot([0, 100], [0, 100], "k--", lw=1)
axes[1].plot([0.5, 3.0], [0.5, 3.0], "k--", lw=1)
axes[0].set_xlabel("True CBF"); axes[0].set_ylabel("Predicted CBF"); axes[0].set_title("Clean CBF")
axes[1].set_xlabel("True ATT"); axes[1].set_ylabel("Predicted ATT"); axes[1].set_title("Clean ATT")
for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "clean_scatter.png"), dpi=120); plt.show()

# 3 & 4. RMSE / CCC vs SNR
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
finite_snr = snr_results_df[np.isfinite(snr_results_df["SNR"])]
for name in MODEL_NAMES:
    sub = finite_snr[finite_snr["Model"] == name].sort_values("SNR")
    axes[0, 0].plot(sub["SNR"], sub["CBF_RMSE"], "o-", label=name)
    axes[0, 1].plot(sub["SNR"], sub["ATT_RMSE"], "o-", label=name)
    axes[1, 0].plot(sub["SNR"], sub["CBF_CCC"], "o-", label=name)
    axes[1, 1].plot(sub["SNR"], sub["ATT_CCC"], "o-", label=name)
for ax, title, ylab in zip(axes.flat,
                            ["CBF RMSE vs SNR", "ATT RMSE vs SNR", "CBF CCC vs SNR", "ATT CCC vs SNR"],
                            ["RMSE", "RMSE", "CCC", "CCC"]):
    ax.set_xscale("log"); ax.set_xlabel("SNR"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "snr_curves.png"), dpi=120); plt.show()

# 5 & 6. Error by CBF range / ATT range
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cbf_pivot = cbf_range_df.pivot(index="Range", columns="Model", values="RMSE")
cbf_pivot.plot(kind="bar", ax=axes[0], legend=False)
axes[0].set_title("CBF error by range"); axes[0].set_ylabel("RMSE"); axes[0].tick_params(axis="x", rotation=45)
att_pivot = att_range_df.pivot(index="Range", columns="Model", values="RMSE")
att_pivot.plot(kind="bar", ax=axes[1], legend=False)
axes[1].set_title("ATT error by range"); axes[1].set_ylabel("RMSE"); axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "error_by_range.png"), dpi=120); plt.show()


# ### 8f. DNN-G vs Bayesian vs NLS — final comparison table
# 
# Bayesian and NLS numbers below are hardcoded from your existing external benchmark run (not computed in this notebook) at SNR=10 and SNR=15, so DNN-G's freshly computed numbers can be checked against them directly, in place, every time this notebook runs.

# In[ ]:


torch.save(CBFnet_G.state_dict(), os.path.join(OUTPUT_DIR, "CBF_G_best.pth"))
torch.save(ATTnet_G.state_dict(), os.path.join(OUTPUT_DIR, "ATT_G_best.pth"))

# Results tables (clean_results, snr_results, cbf/att range were already saved above)

experiment_config = {
    "random_seed": SEED,
    "quick_test_mode": QUICK_TEST,
    "training_samples": {"N_train": N_TRAIN, "N_val": N_VAL, "N_test": N_TEST},
    "cbf_range": [CBF_MIN, CBF_MAX],
    "att_range": [ATT_MIN, ATT_MAX],
    "snr_values": [float(v) if np.isfinite(v) else None for v in SNR_VALUES],
    "dnn_architecture": {
        "CBFnet_G": {"n_hidden_layers": 9, "n_neurons": 50, "params": n_params_cbf},
        "ATTnet_G": {"n_hidden_layers": 9, "n_neurons": 100, "params": n_params_att},
    },
    "loss_function": "MAE (L1Loss)",
    "normalization": {
        "X_mean": X_mean.squeeze().tolist(),
        "X_std": X_std.squeeze().tolist()
    },
    "target_standardization": {
        "CBF_mean": CBF_mean,
        "CBF_std": CBF_std,
        "ATT_mean": ATT_mean,
        "ATT_std": ATT_std,
    },
    "monotonicity_violations": monotonicity_violations,
}

with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w") as f:
    json.dump(experiment_config, f, indent=2)

print(f"All models, tables, plots, and experiment_config.json saved to: {OUTPUT_DIR}")
print("\nFiles written:")
for root, _, files in os.walk(OUTPUT_DIR):
    for fn in sorted(files):
        print("  ", os.path.join(root, fn).replace(OUTPUT_DIR, "").lstrip("/"))


# ## SECTION 9 — Save models, results, and configuration

# In[ ]:


print("============================================================")
print("DNN-G FINAL CLEAN PERFORMANCE")
print("============================================================")
clean_cbf_dnn = clean_results_df[clean_results_df["Model"] == "DNN-G"].iloc[0]
print(f"CBF:\nRMSE: {clean_cbf_dnn['CBF_RMSE']:.4f}\nMAE:  {clean_cbf_dnn['CBF_MAE']:.4f}\nR\u00b2:   {clean_cbf_dnn['CBF_R2']:.4f}\nCCC:  {clean_cbf_dnn['CBF_CCC']:.4f}\n")
print(f"ATT:\nRMSE: {clean_cbf_dnn['ATT_RMSE']:.4f}\nMAE:  {clean_cbf_dnn['ATT_MAE']:.4f}\nR\u00b2:   {clean_cbf_dnn['ATT_R2']:.4f}\nCCC:  {clean_cbf_dnn['ATT_CCC']:.4f}\n")

print("============================================================")
print("DNN-G SNR PERFORMANCE")
print("============================================================")
dnn_snr = snr_results_df[snr_results_df["Model"] == "DNN-G"].sort_values("SNR")
print(f"{'SNR':<6} | {'CBF RMSE':<10} | {'CBF MAE':<10} | {'CBF CCC':<10} | {'ATT RMSE':<10} | {'ATT MAE':<10} | {'ATT CCC':<10}")
print("-" * 85)
for _, row in dnn_snr.iterrows():
    print(f"{row['SNR']:<6} | {row['CBF_RMSE']:<10.4f} | {row['CBF_MAE']:<10.4f} | {row['CBF_CCC']:<10.4f} | {row['ATT_RMSE']:<10.4f} | {row['ATT_MAE']:<10.4f} | {row['ATT_CCC']:<10.4f}")

# ------------------------------------------------------------------
# External reference numbers (Bayesian, NLS) — hardcoded from your
# existing separate benchmark notebook, NOT computed here. Kept in
# one place so every SNR-level comparison below can check DNN-G
# against them without re-typing the numbers each time.
# ------------------------------------------------------------------
EXTERNAL_REFERENCE = {
    10.0: {
        "Bayesian": {"CBF_RMSE": 5.030, "CBF_MAE": 3.968, "CBF_CCC": 0.9606, "ATT_RMSE": 0.0826, "ATT_MAE": 0.0683, "ATT_CCC": 0.9914},
        "NLS":      {"CBF_RMSE": 2.982, "CBF_MAE": 2.048, "CBF_CCC": 0.9913, "ATT_RMSE": 0.5062, "ATT_MAE": 0.3411, "ATT_CCC": 0.6770},
    },
    15.0: {
        "Bayesian": {"CBF_RMSE": 3.759, "CBF_MAE": 2.849, "CBF_CCC": 0.9671, "ATT_RMSE": 0.0584, "ATT_MAE": 0.0472, "ATT_CCC": 0.9944},
        "NLS":      {"CBF_RMSE": 2.062, "CBF_MAE": 1.414, "CBF_CCC": 0.9958, "ATT_RMSE": 0.4951, "ATT_MAE": 0.3259, "ATT_CCC": 0.6899},
    },
}

print("\n============================================================")
print("DNN-G vs BAYESIAN vs NLS (external reference numbers)")
print("============================================================")
for snr, ref in EXTERNAL_REFERENCE.items():
    dnn_row = dnn_snr[dnn_snr["SNR"] == snr]
    print(f"\n--- SNR = {snr} ---")
    print(f"{'Model':<12} | {'CBF RMSE':<9} | {'CBF MAE':<9} | {'CBF CCC':<9} | {'ATT RMSE':<9} | {'ATT MAE':<9} | {'ATT CCC':<9}")
    print("-" * 78)
    if len(dnn_row):
        r = dnn_row.iloc[0]
        print(f"{'DNN-G':<12} | {r['CBF_RMSE']:<9.4f} | {r['CBF_MAE']:<9.4f} | {r['CBF_CCC']:<9.4f} | {r['ATT_RMSE']:<9.4f} | {r['ATT_MAE']:<9.4f} | {r['ATT_CCC']:<9.4f}")
    else:
        print(f"{'DNN-G':<12} | (no exact SNR={snr} row in snr_results_df — check SNR_VALUES)")
    for name, m in ref.items():
        print(f"{name:<12} | {m['CBF_RMSE']:<9.4f} | {m['CBF_MAE']:<9.4f} | {m['CBF_CCC']:<9.4f} | {m['ATT_RMSE']:<9.4f} | {m['ATT_MAE']:<9.4f} | {m['ATT_CCC']:<9.4f}")


# In[ ]:


# ============================================================
# DEFINITIVE SNR=INFINITY vs CLEAN POPULATION CHECK
# ============================================================

print("=" * 75)
print("SNR = INFINITY vs CLEAN INPUT CONSISTENCY CHECK")
print("=" * 75)

# ------------------------------------------------------------
# 1. Direct clean training-distribution input
# ------------------------------------------------------------

X_clean_direct = clean_input_phys.astype(np.float32)

# ------------------------------------------------------------
# 2. SNR = infinity generated input
# ------------------------------------------------------------

sig = clean_sig.astype(np.float32)

e1 = np.zeros_like(sig)
e2 = np.zeros_like(sig)
e3 = np.zeros_like(sig)
e4 = np.zeros_like(sig)

mc = np.sqrt(
    (sig + e1)**2 + e2**2
)

ml = np.sqrt(
    (sig + e3)**2 + e4**2
)

X_clean_snr_inf = (
    mc + ml
).astype(np.float32)

# ------------------------------------------------------------
# 3. Compare
# ------------------------------------------------------------

print("\nDirect clean input:")
print("shape:", X_clean_direct.shape)
print("min  :", X_clean_direct.min())
print("max  :", X_clean_direct.max())
print("mean :", X_clean_direct.mean())

print("\nSNR=inf generated input:")
print("shape:", X_clean_snr_inf.shape)
print("min  :", X_clean_snr_inf.min())
print("max  :", X_clean_snr_inf.max())
print("mean :", X_clean_snr_inf.mean())

difference = X_clean_snr_inf - X_clean_direct

print("\nDifference:")
print("max abs difference :", np.max(np.abs(difference)))
print("mean abs difference:", np.mean(np.abs(difference)))

assert np.allclose(
    X_clean_snr_inf,
    X_clean_direct,
    atol=1e-5
), "SNR=inf does NOT reproduce the direct clean input!"

print("\n✓ SNR=inf exactly reproduces the clean input.")


# In[ ]:


# ============================================================
# DNN CLEAN vs SNR=INFINITY PREDICTION CHECK
# ============================================================

X_direct_norm = (
    X_clean_direct - X_mean
) / X_std

X_inf_norm = (
    X_clean_snr_inf - X_mean
) / X_std

pred_direct = predict_all_models(X_direct_norm)
pred_inf = predict_all_models(X_inf_norm)

for name in MODEL_NAMES:

    print("\n" + "=" * 60)
    print(name)

    cbf_diff = (
        pred_inf[name]["CBF"]
        - pred_direct[name]["CBF"]
    )

    att_diff = (
        pred_inf[name]["ATT"]
        - pred_direct[name]["ATT"]
    )

    print(
        "CBF max prediction difference:",
        np.max(np.abs(cbf_diff))
    )

    print(
        "ATT max prediction difference:",
        np.max(np.abs(att_diff))
    )

    m_direct_cbf = calc_metrics(
        clean_cbf_true,
        pred_direct[name]["CBF"]
    )

    m_inf_cbf = calc_metrics(
        clean_cbf_true,
        pred_inf[name]["CBF"]
    )

    m_direct_att = calc_metrics(
        clean_att_true,
        pred_direct[name]["ATT"]
    )

    m_inf_att = calc_metrics(
        clean_att_true,
        pred_inf[name]["ATT"]
    )

    print("\nCBF")
    print(
        "Direct clean RMSE:",
        m_direct_cbf["RMSE"]
    )
    print(
        "SNR=inf RMSE     :",
        m_inf_cbf["RMSE"]
    )

    print("\nATT")
    print(
        "Direct clean RMSE:",
        m_direct_att["RMSE"]
    )
    print(
        "SNR=inf RMSE     :",
        m_inf_att["RMSE"]
    )


# In[ ]:




