# Architecture 01 Pre-Architecture-02 Audit

**Date:** 2026-09-29  
**Status:** AUDIT COMPLETE ΓÇö Issues found and classified

---

## A. GitHub State

**Finding:** Architecture 01 files ARE committed to the local `HEAD` (commit `aed1c4b`) but are **NOT** present in `origin/main` (commit `870854d`).

The local `main` branch is **1 commit ahead** of `origin/main`.

The push failed silently with `HTTP 408 Timeout` during large-file upload (PNGs and CSVs). The commit was created locally but never reached GitHub.

`origin/main` contains only these Arch01-related files: **none**.  
`origin/main` does NOT contain:
- `architectures/01_shared_kinetic_representation/`
- `comparisons/architecture_00_vs_01.csv`

**Action required:** Push the pending local commit (`aed1c4b`) to origin.

---

## B. Local State

Local `HEAD` (`aed1c4b`) contains:

```
architectures/01_shared_kinetic_representation/
    README.md          Γ£ô
    architecture.md    Γ£ô
    changes.md         Γ£ô
    config.yaml        Γ£ô
    evaluate.py        Γ£ô
    generate_csvs.py   Γ£ô
    model.py           Γ£ô
    results.md         Γ£ô
    train.py           Γ£ô
    figures/
        att_response.png              Γ£ô
        cbf_monotonicity.png          Γ£ô
        shared_latent_pca_att.png     Γ£ô
        shared_latent_pca_cbf.png     Γ£ô
    tables/
        error_by_att_range.csv        Γ£ô
        error_by_cbf_range.csv        Γ£ô
        raw_results.csv               Γ£ô
comparisons/architecture_00_vs_01.csv  Γ£ô
```

**Additional finding:** Several source files were modified after the last commit and are currently unstaged:
- `architectures/01_shared_kinetic_representation/*.py` ΓÇö modified (CRLF warnings from git reset)
- `src/data/generate_dataset.py` ΓÇö modified
- `src/data/normalization.py` ΓÇö modified
- `src/models/common.py` ΓÇö modified
- `src/training/evaluation.py` ΓÇö modified

The local working copies contain the correct canonical content. The modifications vs HEAD are cosmetic (CRLF encoding differences from `git rm --cached / git add`). No substantive code was changed.

---

## C. Architecture 01 Implementation Audit

### model.py ΓÇö Matches documentation
- SharedKineticNet: encoder 6├ù100 ELU Γ£ô
- CBF head: 3├ù50 ELU + output Γ£ô
- ATT head: 3├ù100 ELU + output Γ£ô
- Kaiming init Γ£ô
- `predict_physical` un-standardises and clamps Γ£ô
- Output: `(torch.stack([cbf_std, att_std], dim=-1), latent)` Γ£ô

### train.py ΓÇö Partially matches documentation

| Aspect | Documented | Actual |
|--------|-----------|--------|
| Optimizer | Adam LR=1e-3 | Γ£ô |
| Batch size | 512 | Γ£ô |
| Max epochs | 30 (smoke test) | Γ£ô |
| Patience | 10 | Γ£ô |
| Train N | 20,000 | Γ£ô |
| Val N | 5,000 | Γ£ô |
| Seeds | 42, 123, 2024 | Γ£ô |
| Loss | MAE (L1Loss) | Γ£ô |
| Grad clip | 1.0 | Γ£ô |

**Issue:** The baseline inside `train.py` is `BaselineComboNet` which uses **9├ù50 for CBF** and **9├ù100 for ATT** with `StandardizedNet`. But `StandardizedNet` in `src/models/common.py` outputs a **single scalar** and takes `y_mean`, `y_std`, `y_min`, `y_max` **per parameter**. The `BaselineComboNet.forward()` calls both sub-nets and stacks them. This is architecturally correct. However, the baseline networks inside `train.py` were trained **on the same data as Arch01**, which is the right experimental design ΓÇö they are not the original paper-matched models. This is correctly documented in `results.md` as a smoke-test comparison only.

**Issue:** `generate_csvs.py` (separate evaluation script) re-generates normalizer from a second `generate_data(20_000, rng)` call with a fresh `rng` (seeded from the same seed). This means the normalizer used during evaluation **may not exactly match** the normalizer used during training, because `rng` was advanced by the `get_data()` call that consumed a dummy `train_loader` first. This is a mild procedural flaw ΓÇö the normalizer statistics will be very close but not guaranteed identical. It does not affect relative rankings.

### evaluate.py ΓÇö Contains correct diagnostics
- ATT-range error decomposition Γ£ô
- CBF-range error decomposition Γ£ô
- Monotonicity and ATT-response plots Γ£ô
- Signal reconstruction diagnostic Γ£ô
- Latent PCA analysis Γ£ô
- Latent-dimension correlations Γ£ô

### config.yaml ΓÇö Correct Γ£ô

### results.md ΓÇö Correct in structure, see noise audit below

---

## D. Noise Pipeline Audit

### The complete signal path

```
True CBF, ATT
    Γåô
compute_signals_vec(cbf, att) * SCALE  ΓåÆ  S   (true clean ASL signal)
    Γåô
add_noise(S, noise_sd)  ΓåÆ  X
    = sqrt((S+e1)┬▓ + e2┬▓) + sqrt((S+e3)┬▓ + e4┬▓)
    Γåô
Normalization  ΓåÆ  X_norm
    Γåô
Network(X_norm)  ΓåÆ  z_CBF, z_ATT
    Γåô
Un-standardize  ΓåÆ  CBF_pred, ATT_pred
    Γåô
Evaluation: calc_metrics(CBF_true, CBF_pred)
```

### The 2├ù scaling fact

When `noise_sd = 0` exactly:

```
X = sqrt(S┬▓ + 0) + sqrt(S┬▓ + 0) = S + S = 2S
```

This is **mathematically correct behaviour** of the dual-Rician magnitude model at zero noise. It is **not a bug**. It reflects the fact that the acquisition model sums two independent magnitude images.

### Where 2├ù appears in each dataset

| Dataset | How generated | Signal content |
|---------|--------------|----------------|
| **Training** | `generate_data(N, rng, n_noise_levels=100)` | 1% samples have `sd=0` ΓåÆ `X=2S`; 99% have `sd>0` ΓåÆ `X=Rician(S,sd)+Rician(S,sd)` |
| **Clean eval** | `generate_data(N, rng, n_noise_levels=0)` | 100% `sd=0` ΓåÆ `X=2S` always |
| **Noisy eval (generate_csvs.py)** | `X_ts = add_noise(X_ts_clean, sd)` | **PROBLEM: applies `add_noise` to `X=2S` not to `S`** |

### Is normalization affected?

**Training normalizer** is fit on data where X is dominated by the Rician sum:
- `X_train_mean[PLD0] Γëê 660`
- `S_mean[PLD0] Γëê 324` (roughly 2├ù)

The training data correctly includes the Rician floor even at zero noise. The mean of the training data is roughly `2├ùS` because that is what the dual-Rician model produces even at moderate noise.

**Measured distribution shift (clean eval vs train):** 0.02ΓÇô0.04 standard deviations.  
This is **very small** and does not materially affect the comparison.

### The noisy evaluation bug

**In `generate_csvs.py`:**
```python
X_ts_clean, Y_ts, _ = generate_data(5_000, rng_test, n_noise_levels=0)
# X_ts_clean = 2S  ΓåÉ already processed

for snr in [inf, 50, 20, 15, 10, 5]:
    X_ts = add_noise(X_ts_clean, sd, ...)
    # This computes: sqrt((2S + e)┬▓ + e┬▓) + sqrt((2S + e)┬▓ + e┬▓)
    # NOT: sqrt((S + e)┬▓ + e┬▓) + sqrt((S + e)┬▓ + e┬▓)
```

The noisy evaluations effectively apply the Rician noise to `2S` instead of `S`. This **doubles the effective signal amplitude** before noise is added, meaning the actual SNR experienced is approximately **twice** the intended SNR.

| Intended SNR | Actual effective SNR (approx) |
|---|---|
| 5 | ~10 |
| 10 | ~20 |
| 20 | ~40 |
| 50 | ~100 |
| Γê₧ | Γê₧ (2S, consistent) |

**All three models (Baseline, Arch01, Ablation) experienced this identically**, so the **relative comparison is preserved**. However, the **absolute RMSE values at each SNR label are wrong** ΓÇö they describe a cleaner scenario than the label implies.

---

## E. Evaluation Validity

| Result type | Valid for relative comparison? | Valid absolute values? |
|---|---|---|
| Clean (SNR=Γê₧) results | **Yes** | **Approximate** (small ~2ΓÇô4% normalizer shift, but all consistent) |
| SNR 50/20/15/10/5 relative comparison | **Yes** (all models same pipeline) | **No** (effective SNR ~2├ù higher than label) |
| ATT-regime error breakdown | **Yes** (clean data, consistent) | **Approximate** |
| CBF-range error breakdown | **Yes** (clean data, consistent) | **Approximate** |
| Latent PCA / correlations | **Yes** (diagnostic only) | N/A |
| Signal reconstruction diagnostic | **Yes** | **Yes** |

---

## F. Scientific Status of Architecture 01

### What is scientifically established

1. **Under this smoke-test configuration (20k samples, 30 epochs, 3 seeds), shared encoding produced negligibly different performance compared to a capacity-matched independent network.**

2. The shared latent space demonstrably encodes both CBF (max latent correlation 0.90) and ATT (max latent correlation 0.86) ΓÇö the shared encoder works as intended mechanistically.

3. ATT estimation is worst in the `0.5ΓÇô1.0 s` regime (RMSE ~0.38 s) and best in `1.5ΓÇô1.8 s` (RMSE ~0.10 s), consistent with the information analysis.

### What is NOT established

1. That shared encoding provides **no benefit** ΓÇö the smoke test is insufficiently powered to detect small differences.
2. That the **absolute** noisy-SNR RMSE values are accurate ΓÇö they are not.
3. That the result generalises to the full training regime (200k samples, convergence to plateau).

### Corrected conclusion

> **Under the controlled smoke-test configuration (20k training samples, 30 epochs, seeds 42/123/2024), a shared encoder + parallel CBF/ATT heads produced no measurable improvement over a capacity-matched pair of independent networks. The result is consistent with the hypothesis that generic weight sharing, without structural inductive bias, does not exploit the kinetic structure of the ASL forward model. This conclusion should be treated as preliminary pending full-scale training.**

---

## G. Files Requiring Rerun

| File/result | Action needed |
|---|---|
| `comparisons/architecture_00_vs_01.csv` | Rerun `generate_csvs.py` with corrected noisy-eval pipeline (apply `add_noise` to `S_true`, not `X_clean`) |
| `tables/raw_results.csv` | Same |
| All SNR columns in `results.md` | Update after rerun |
| Clean (SNR=Γê₧) results | **Usable as-is for relative comparison** |
| ATT/CBF-range tables | **Usable as-is** (generated from clean pipeline) |
| Figures | **Usable as-is** |

The **corrected noisy evaluation** should use:
```python
S_true = compute_signals_vec(cbf_gt, att_gt) * SCALE
for snr in [...]:
    X_ts = add_noise(S_true, 334.2039/snr, rng=...)
```
not chain `add_noise(add_noise(...))`.

---

## H. What Is Safe to Use as Evidence for Architecture 02

The following are safe to cite:

1. **Clean RMSE comparison (Arch01 vs Baseline vs Ablation):** All are approximately equal. Weight sharing alone does not improve parameter estimation.
2. **ATT-regime error breakdown:** Short ATTs are hardest; PLD-straddling ATTs are easiest. Consistent with theory.
3. **Latent representation analysis:** Shared encoder learns structured representations of both parameters.
4. **Information analysis results (Phase 1ΓÇô2):** Sound and unaffected.
5. **The corrected conclusion** stated in Section F above.

**Do not cite** the per-SNR absolute RMSE values from `raw_results.csv` until the noisy evaluation pipeline is fixed.
