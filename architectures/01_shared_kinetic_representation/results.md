# Architecture 01 Results

> **Audit note (2026-09-29):** This file was updated after a pipeline audit identified a bug in the
> noisy-SNR evaluation (previous version applied `add_noise` to `X=2S` rather than to `S_true`,
> doubling effective SNR). The corrected numbers below come from `generate_csvs.py` (fixed version).
> See `analysis/architecture_01_audit.md` for the full diagnosis.

---

## 1. Configuration

| Setting | Value |
|---|---|
| Experiment type | Smoke test |
| Training samples | 20,000 |
| Validation samples | 5,000 |
| Test samples | 5,000 (seed 7) |
| Seeds | 42, 123, 2024 |
| Max epochs | 30 |
| Patience | 10 |
| Optimizer | Adam (LR = 1e-3) |
| Batch size | 512 |
| Loss | MAE (L1) on standardized targets |

## 2. Parameter Count

| Model | Parameters |
|---|---|
| Baseline (9×50 CBF + 9×100 ATT, independent) | 102,102 |
| Architecture 01 (6×100 shared + 3×50 CBF head + 3×100 ATT head) | 91,602 |
| Ablation (independent, reduced 9×70 + 9×70) | 80,362 |

## 3. Training Behaviour

All models converged stably across 3 seeds. Architecture 01 reached best validation loss marginally faster than the Baseline but achieved similar final validation MAE (~0.23–0.24 in standardized units).

## 4. Clean Performance (SNR = ∞, corrected)

*Mean ± std across 3 seeds, 5,000 test samples, correct clean pipeline (`X=2S`, consistent with training).*

| Model | CBF RMSE | CBF MAE | ATT RMSE | ATT MAE |
|---|---:|---:|---:|---:|
| Baseline | 3.09 ± 0.09 | — | 0.264 ± 0.010 | — |
| Ablation | 2.92 ± 0.24 | — | 0.257 ± 0.003 | — |
| **Architecture 01** | **2.96 ± 0.19** | — | **0.259 ± 0.011** | — |

Differences are within the standard deviation across seeds. No model is clearly superior under the smoke-test regime.

## 5. SNR Performance (corrected pipeline)

*Mean CBF RMSE across 3 seeds. SNR is now correctly computed: `add_noise(S_true, 334.2/SNR)`.*

| SNR | Baseline CBF RMSE | Arch01 CBF RMSE | Ablation CBF RMSE |
|---|---:|---:|---:|
| ∞ (clean) | 3.09 | 2.96 | 2.92 |
| 50 | 3.14 | 3.01 | 2.97 |
| 20 | 3.42 | 3.36 | 3.29 |
| 15 | 3.68 | 3.65 | 3.57 |
| 10 | 4.36 | 4.39 | 4.28 |
| 5 | 7.68 | 7.70 | 7.62 |

| SNR | Baseline ATT RMSE | Arch01 ATT RMSE | Ablation ATT RMSE |
|---|---:|---:|---:|
| ∞ (clean) | 0.264 | 0.259 | 0.258 |
| 50 | 0.275 | 0.270 | 0.271 |
| 20 | 0.307 | 0.305 | 0.311 |
| 15 | 0.329 | 0.328 | 0.335 |
| 10 | 0.375 | 0.375 | 0.381 |
| 5 | 0.496 | 0.491 | 0.496 |

**Observation:** All three models are statistically indistinguishable within the smoke-test variance. No model demonstrates consistent dominance across all SNR levels and seeds.

## 6. CBF-range Performance

*(From Arch01 Seed 42, dense clean test)*

| CBF Range | CBF RMSE | ATT RMSE |
|---|---:|---:|
| 0–20 | ~2.1 | ~0.40 |
| 20–40 | ~2.4 | ~0.18 |
| 40–60 | ~2.4 | ~0.18 |
| 60–80 | ~3.0 | ~0.14 |
| 80–100 | ~3.5 | ~0.13 |

ATT estimation improves at higher CBF (higher SNR signal). CBF estimation degrades at extremes.

## 7. ATT-range Performance

*(From Arch01 Seed 42, dense clean test)*

| ATT Range | CBF RMSE | ATT RMSE | Information-analysis prediction |
|---|---:|---:|---|
| 0.5–1.0 | ~4.2 | ~0.38 | High ambiguity (scale-like regime) |
| 1.0–1.4 | ~2.8 | ~0.22 | Moderate |
| 1.4–1.525 | ~2.0 | ~0.12 | Approaching first PLD |
| 1.525–1.8 | ~1.9 | ~0.10 | Best conditioned (straddling PLDs) |
| 1.8–2.2 | ~2.1 | ~0.13 | Good |
| 2.2–2.6 | ~2.1 | ~0.16 | Moderate |
| 2.6–3.0 | ~2.2 | ~0.28 | Signal decay region |

This is **fully consistent with the information analysis**. The inverse problem is hardest at short ATTs (< 1.0 s) and best conditioned when ATT straddles the measurement PLDs.

## 8. Latent Representation Analysis

- PCA of 100-dimensional shared latent space shows smooth stratification by both CBF and ATT.
- Max linear correlation of any single latent neuron with true CBF: **0.903**
- Max linear correlation of any single latent neuron with true ATT: **0.860**
- Signal reconstruction error (diagnostic, not loss): Arch01 = 510.0, Baseline = 509.5 (essentially identical)

**The shared encoder successfully embeds both parameters into structured latent directions.** However, this does not improve prediction.

## 9. Baseline Comparison

| Metric | Baseline | Architecture 01 | Difference |
|---|---:|---:|---:|
| CBF RMSE (clean) | 3.09 | 2.96 | −0.13 (−4%) |
| ATT RMSE (clean) | 0.264 | 0.259 | −0.005 (−2%) |
| CBF RMSE (SNR 10) | 4.36 | 4.39 | +0.03 (+1%) |
| ATT RMSE (SNR 10) | 0.375 | 0.375 | ~0 |

Differences are within the range of seed-to-seed variability. No consistent improvement.

## 10. Sharing Ablation

The Independent Ablation network (~80k params, 9×70 each) matches Arch01 in clean performance and is occasionally better at some seeds. **This confirms the performance is driven by model capacity, not by weight sharing.**

## 11. Multi-seed Results (corrected)

| Metric | Baseline | Architecture 01 | Ablation |
|---|---:|---:|---:|
| CBF RMSE (clean) mean | 3.09 | 2.96 | 2.92 |
| CBF RMSE (clean) std | 0.09 | 0.19 | 0.24 |
| ATT RMSE (clean) mean | 0.264 | 0.259 | 0.258 |
| ATT RMSE (clean) std | 0.010 | 0.011 | 0.003 |

## 12. Interpretation

**OBSERVED RESULT:** Under the smoke-test regime, generic weight sharing produced no measurable improvement over a capacity-matched independent network.

**INTERPRETATION:** The shared encoder demonstrably learns to embed both CBF and ATT into structured latent directions (latent correlations > 0.86). However, since the encoder and heads are all generic MLPs, the network has no mechanism to exploit the *physical relationship* between the two parameters: namely that ATT dictates the kinetic shape, and CBF is a terminal linear scaler on that shape.

**HYPOTHESIS:** Generic weight sharing is an insufficient inductive bias for this problem. What is needed is a structural prior that encodes the *asymmetric dependency* — ATT governs shape, CBF scales amplitude.

## 13. Limitations

1. **Smoke test only:** 20k samples, 30 epochs. The experiment is underpowered to detect small improvements; it can only detect large effects or confirm approximate equality.
2. **Normalizer re-generation:** The evaluation script re-generates normalizer statistics from a fresh data draw, which may introduce small but non-zero differences from the training normalizer.
3. **The 2S property:** Clean evaluation inputs are `X = 2S` (dual-Rician at zero noise), which is ~1–4% shifted from the training distribution mean. This is a property of the physical model, not a bug, but must be acknowledged.

## 14. Decision for Architecture 02

**Decision: REJECT generic sharing.**

**Rationale:** Generic weight sharing (parallel CBF and ATT heads on a shared encoder) is equivalent in performance to a capacity-matched pair of independent networks under our smoke-test conditions. The improvement ceiling from generic sharing appears to be negligible (<5% RMSE).

**Next hypothesis for Architecture 02:** Hierarchical kinetic conditioning — predict ATT first, then condition CBF estimation on the extracted kinetic representation, mirroring the physical dependency $\text{CBF} \propto S / f(\text{ATT})$.
