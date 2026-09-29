# Architecture 03 Results
## Gradient-Isolated Kinetic Conditioning

**Experiment ID:** Arch03-GradientIsolated
**Date:** 2026-09-29

### Configuration
*   **Dataset:** 20,000 train, 5,000 val, 5,000 test.
*   **Architecture:** GradientIsolatedKineticNet (94,772 params).
*   **Setup:** Identical to Architecture 02 (Adam, 1e-3, MAE loss, seeds 42, 123, 2024, identical SNRs).

### Gradient Flow Verification
An explicit backward pass test confirmed the implementation functions exactly as intended:
*   CBF gradient into kinetic_encoder = **0.0000** (Gradient successfully stopped)
*   Representation mathematical dependence = **1.3847** (Representation still influences CBF)

### Latent Representation Purity
| Model | Max Corr(z, CBF) | Max Corr(z, ATT) | R² Probe(CBF) | R² Probe(ATT) |
| :--- | :--- | :--- | :--- | :--- |
| **Arch 02** | 0.915 | 0.862 | 0.993 | 0.912 |
| **Arch 03** | 0.825 | 0.854 | 0.993 | 0.909 |

*Finding:* Isolating the gradient successfully reduced the direct CBF correlation inside the kinetic representation (0.915 -> 0.825), validating that Arch 02 was "polluting" the timing prior with amplitude information. However, the overall linear probe R² remained very high, indicating the required information remains fundamentally coupled in the shared latent space.

### Overall SNR Robustness (Mean CBF RMSE)
| SNR | Arch 02 CBF | Arch 03 CBF | Δ (A03 - A02) |
| :--- | :--- | :--- | :--- |
| inf | 2.931 | 3.038 | +0.107 (worse) |
| 50 | 2.986 | 3.089 | +0.103 (worse) |
| 20 | 3.311 | 3.410 | +0.099 (worse) |
| 15 | 3.593 | 3.697 | +0.104 (worse) |
| 10 | 4.303 | 4.431 | +0.128 (worse) |
| 5 | 7.557 | 7.657 | +0.100 (worse) |

### ATT Regime Error Breakdown (inf SNR)
| ATT Bin (s) | Arch 02 CBF | Arch 03 CBF | Δ (A03 - A02) | Arch 02 ATT | Arch 03 ATT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.5-1.0 | 3.685 | 3.501 | **-0.184 (better)** | 0.429 | 0.382 |
| 1.0-1.5 | 3.753 | 4.031 | +0.278 (worse) | 0.244 | 0.274 |
| 1.5-2.0 | 2.751 | 3.217 | +0.466 (worse) | 0.201 | 0.228 |
| 2.0-2.5 | 1.322 | 2.067 | +0.745 (worse) | 0.075 | 0.087 |
| 2.5-3.0 | 1.749 | 2.312 | +0.563 (worse) | 0.180 | 0.193 |

### Interpretation & Conclusion
The experiment demonstrates a fascinating trade-off. By blocking CBF gradients from shaping the kinetic representation, the overall network performance degraded (mean CBF RMSE rose by ~0.11 across all SNRs). This indicates that the joint optimization of z_kin by both objectives in Architecture 02 is mutually beneficial for generalized estimation.

However, in the **most severely ambiguous regime (ATT 0.5–1.0s)**, the gradient-isolated Architecture 03 actually **outperformed** Architecture 02, improving both CBF (3.68 -> 3.50) and ATT (0.43 -> 0.38) predictions. This suggests that a strictly pure kinetic prior is uniquely powerful at breaking the hardest shape/scale ambiguities, even if it is slightly less optimal for standard regimes.

**Scientific Outcome:** Gradient isolation proved that joint optimization generally aids representation learning, but strict priors remain superior at the physical boundary limits of the forward model.

### Corrected Latent Representation Analysis (Linear Probe)
A subsequent rigorous evaluation of z_kin was performed using a true linear probe (trained on 20,000 independent samples normalized with the canonical noisy configuration, and tested on the 5,000 held-out test set).

*   **Purity Shift:** Gradient isolation definitively made individual latent dimensions **more ATT-specific** (median correlation rose from 0.43 to 0.50) and **less CBF-sensitive** (median correlation dropped from 0.48 to 0.34).
*   **Information Content:** The overall linearly recoverable CBF information in z_kin slightly degraded under Architecture 03 (Test RMSE 3.11 vs 2.97).
*   **Mechanistic Explanation:** The localized network improvement in the short-ATT (0.5–1.0s) regime is not because z_kin contains *more* CBF information (the linear probe CBF RMSE is actually worse there). Instead, the improvement is likely due to the non-linear CBF head benefiting from a stricter timing prior that isn't contorted by conflicting amplitude gradients, effectively allowing it to break the shape/scale ambiguity. Conversely, the overall performance degradation outside this regime is explained by the loss of these mutually beneficial, CBF-sensitive joint features in the latent space.
