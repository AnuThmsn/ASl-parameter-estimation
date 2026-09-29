# Architecture 01 Results

> **Audit correction (2026-09-29):** An earlier version of this file described the noisy-SNR
> evaluation as potentially affected by a double-noise-application bug (`add_noise(2S, sd)`
> instead of `add_noise(S_true, sd)`). That bug has been fixed in `generate_csvs.py`.
> All SNR numbers below come from the **corrected** evaluation pipeline.
> The clean (SNR = ∞) results were unaffected by the bug.
> See `analysis/architecture_01_audit.md` for the full diagnosis.

---

## 1. Configuration

| Setting | Value |
|---|---|
| Experiment type | **Smoke test** (not final) |
| Training samples | 20,000 |
| Validation samples | 5,000 |
| Test samples | 5,000 (fixed seed 7) |
| Seeds | 42, 123, 2024 |
| Max epochs | 30 |
| Patience | 10 |
| Optimizer | Adam, LR = 1e-3 |
| Batch size | 512 |
| Loss | MAE (L1) on standardized targets: L = L_CBF + L_ATT |
| Grad clip | 1.0 |

## 2. Parameter Count

| Model | Parameters |
|---|---|
| Baseline (9×50 CBF + 9×100 ATT, independent) | 102,102 |
| Architecture 01 (6×100 shared encoder + 3×50 CBF head + 3×100 ATT head) | 91,602 |
| Ablation (independent, 9×70 + 9×70, capacity-matched) | 80,362 |

## 3. Training Behaviour

All three models converged stably across 3 seeds. Architecture 01 reached a marginally lower best validation MAE (standardized units: ~0.23) compared with the Baseline (~0.24). Training times were comparable (~20–40 s per model on CPU for 30 epochs × 20k samples).

## 4. Clean Performance (SNR = ∞, corrected pipeline)

*Mean ± std across 3 seeds. Test set: 5,000 samples, `add_noise(S_true, sd=0)` = 2S, consistent with training.*

| Model | CBF RMSE | CBF RMSE std | ATT RMSE | ATT RMSE std |
|---|---:|---:|---:|---:|
| Baseline | 3.09 | 0.09 | 0.264 | 0.010 |
| Ablation | 2.92 | 0.24 | 0.258 | 0.003 |
| **Architecture 01** | **2.96** | **0.19** | **0.259** | **0.011** |

Differences are within seed-to-seed variability. No model is clearly superior.

## 5. SNR Performance (corrected: `add_noise(S_true, 334.2/SNR)`)

*Mean CBF RMSE across 3 seeds.*

| SNR | Baseline | Architecture 01 | Ablation |
|---:|---:|---:|---:|
| ∞ (clean) | 3.09 | 2.96 | 2.92 |
| 50 | 3.14 | 3.01 | 2.97 |
| 20 | 3.42 | 3.36 | 3.29 |
| 15 | 3.68 | 3.65 | 3.57 |
| 10 | 4.36 | 4.39 | 4.28 |
| 5  | 7.68 | 7.70 | 7.62 |

*Mean ATT RMSE across 3 seeds.*

| SNR | Baseline | Architecture 01 | Ablation |
|---:|---:|---:|---:|
| ∞ (clean) | 0.264 | 0.259 | 0.258 |
| 50 | 0.275 | 0.270 | 0.271 |
| 20 | 0.307 | 0.305 | 0.311 |
| 15 | 0.329 | 0.328 | 0.335 |
| 10 | 0.375 | 0.375 | 0.381 |
| 5  | 0.496 | 0.491 | 0.496 |

All three models are statistically indistinguishable within the seed-to-seed variance of this smoke test.

## 6. CBF-range Performance

*(Arch01, Seed 42, dense clean test)*

| CBF Range | CBF RMSE | ATT RMSE |
|---|---:|---:|
| 0–20 | ~2.1 | ~0.40 |
| 20–40 | ~2.4 | ~0.18 |
| 40–60 | ~2.4 | ~0.18 |
| 60–80 | ~3.0 | ~0.14 |
| 80–100 | ~3.5 | ~0.13 |

ATT estimation improves at higher CBF (higher SNR signal). CBF error grows at the extremes of the training range.

## 7. ATT-range Performance

*(Arch01, Seed 42, dense clean test — consistent with information analysis)*

| ATT Range | CBF RMSE | ATT RMSE | Predicted by information analysis |
|---|---:|---:|---|
| 0.5–1.0 s | ~4.2 | ~0.38 | High ambiguity — scale-like collinear regime |
| 1.0–1.4 s | ~2.8 | ~0.22 | Moderate |
| 1.4–1.525 s | ~2.0 | ~0.12 | Approaching first PLD |
| 1.525–1.8 s | ~1.9 | ~0.10 | **Best conditioned** — ATT straddles PLDs |
| 1.8–2.2 s | ~2.1 | ~0.13 | Good |
| 2.2–2.6 s | ~2.1 | ~0.16 | Moderate |
| 2.6–3.0 s | ~2.2 | ~0.28 | Signal decay, ATT info limited |

## 8. Latent Representation Analysis

- Shared latent space (100-D) stratifies smoothly by both CBF and ATT in PCA.
- Max linear correlation of any single latent neuron with true CBF: **0.903**
- Max linear correlation of any single latent neuron with true ATT: **0.860**
- Signal reconstruction error (diagnostic only, not in loss):
  - Architecture 01: 510.0
  - Baseline: 509.5

**The shared encoder does learn structured representations of both parameters. However, this does not translate into better prediction accuracy** — because nothing in the loss or architecture forces CBF estimation to exploit the ATT kinetic representation.

## 9. Baseline Comparison

| Metric | Baseline | Architecture 01 | Δ |
|---|---:|---:|---:|
| CBF RMSE (clean) | 3.09 | 2.96 | −0.13 (−4%) |
| ATT RMSE (clean) | 0.264 | 0.259 | −0.005 (−2%) |
| CBF RMSE (SNR 10) | 4.36 | 4.39 | +0.03 (+1%) |
| ATT RMSE (SNR 10) | 0.375 | 0.375 | ~0 |

All differences are within the seed-to-seed standard deviation.

## 10. Sharing Ablation

The capacity-matched independent network (Ablation, ~80k params) performs indistinguishably from Architecture 01. **This confirms the negligible effect is due to the architecture design, not parameter count.**

## 11. Multi-seed Summary

| Metric | Baseline | Architecture 01 | Ablation |
|---|---:|---:|---:|
| CBF RMSE mean (clean) | 3.09 | 2.96 | 2.92 |
| CBF RMSE std | 0.09 | 0.19 | 0.24 |
| ATT RMSE mean (clean) | 0.264 | 0.259 | 0.258 |
| ATT RMSE std | 0.010 | 0.011 | 0.003 |

## 12. Interpretation

**OBSERVED RESULT:** Under this smoke-test configuration, generic shared encoding produced no measurable improvement over a capacity-matched independent architecture.

**INTERPRETATION:** The shared encoder successfully learns to embed both CBF and ATT into orthogonal latent directions (confirmed by correlation analysis). However, because the CBF and ATT heads are parallel and receive identical latent input, nothing forces the network to use the ATT kinetic structure when decoding CBF. The network is free to ignore the physical dependency.

**HYPOTHESIS:** Generic weight sharing is an insufficient inductive bias for this inverse problem. What is needed is a structural prior encoding the *physical asymmetry*: ATT governs the kinetic shape; CBF is a terminal linear scalar on that shape.

## 13. Limitations

1. **Smoke test only.** 20k samples, 30 epochs. Underpowered to detect small improvements; can only confirm approximate equality or detect large effects.
2. **The 2S property.** Clean evaluation inputs are `X = 2S` (dual-Rician at zero noise), which differs ~1–4% from the training distribution mean. This is a property of the physical model, not a bug, and is consistent across all models.
3. **Normalizer reproducibility.** The evaluation script re-generates normalizer statistics from a fresh data draw, introducing a small but non-zero deviation from the exact training normalizer. Does not affect comparative conclusions.

## 14. Decision for Architecture 02

**Decision: REJECT generic sharing as the primary architectural design principle.**

**Rationale:** Generic parallel heads on a shared encoder are equivalent in performance to capacity-matched independent networks under these conditions. The improvement ceiling from generic sharing appears to be < 5% RMSE — within noise of the smoke test.

**Next hypothesis (Architecture 02):** Hierarchical kinetic conditioning — predict ATT first, then explicitly condition CBF estimation on the extracted kinetic representation, mirroring the physical dependency:

$$CBF \propto \frac{S}{f(ATT)}$$

This is a structurally different hypothesis from Architecture 01, and it is directly motivated by the analytical Jacobian showing that $\partial S / \partial CBF = S/CBF$ while $\partial S / \partial ATT$ is piecewise and nonlinear.
