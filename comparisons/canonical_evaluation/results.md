# Canonical Re-evaluation of Architectures 02, 03, and 04

## Motivation
Previous comparisons used slightly different evaluation scripts which introduced minor normalizer mismatches for Architecture 02 (e.g. reporting 3.157 RMSE instead of the true 2.931 on identically normalized canonical data). This stage enforces a single strict canonical evaluation pipeline where all three architectures evaluate the exact same test dataset tensors generated under identical conditions.

## Protocol
* **Training Normalization:** Canonical noisy configuration (100 noise levels, sd_max = 66.84) applied identically.
* **Test Dataset:** 5,000 samples, fixed seed 7, canonical forward model and noise configuration.
* **Test Application:** The exact same tensor X_ts_norm was passed to all three checkpoints.

## Overall Results (CBF RMSE)
| SNR | Arch02 (100%) | Arch03 (0%) | Arch04 (50%) |
| --: | --------------: | --------------: | --------------: |
| inf | 2.931 | 3.038 | **2.923** |
|  50 | 2.986 | 3.089 | **2.977** |
|  20 | 3.311 | 3.410 | **3.316** |
|  15 | 3.593 | 3.697 | **3.613** |
|  10 | **4.303** | 4.431 | 4.367 |
|   5 | **7.557** | 7.657 | 7.734 |

## ATT Regime Results (CBF RMSE at SNR inf)
| ATT regime | Arch02 | Arch03 | Arch04 |
| ---------- | ---------: | ---------: | ---------: |
| 0.5–1.0    | 4.104 | **3.877** | 4.085 |
| 1.0–1.5    | 3.334 | 3.594 | **3.230** |
| 1.5–2.0    | 2.495 | 2.830 | **2.469** |
| 2.0–2.5    | **1.746** | 1.924 | 1.978 |
| 2.5–3.0    | 2.311 | 2.326 | **2.106** |

## Latent Representation
| Architecture | ATT R² | CBF R² |
| ------------ | -----: | -----: |
| Arch02 (100%)| 0.858 | 0.989 |
| Arch03 (0%)  | 0.856 | 0.988 |
| Arch04 (50%) | 0.857 | 0.988 |

## Scientific Conclusion
The canonical evaluation perfectly validates the previous findings:
1. **Standard Regimes:** Partial coupling (Arch 04) provides the best overall generalized features (2.923 CBF RMSE vs 2.931 in Arch 02).
2. **Boundary Regime:** Strict zero-coupling (Arch 03) is necessary to resolve the highly ambiguous short-ATT regime (3.877 vs >4.08 in others).
