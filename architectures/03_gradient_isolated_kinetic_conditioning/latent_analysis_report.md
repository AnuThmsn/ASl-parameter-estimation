========================================
CORRECTED LATENT ANALYSIS AUDIT
========================================

Normalization:
PASS (Noisy configuration exact match)

Test-set isolation:
PASS (Used separate test seed 7)

Probe train/test separation:
PASS (Trained on canonical 20k train set, evaluated on 5k test set)

Checkpoint availability:
Available locally. (Note: *.pt is ignored in .gitignore, so exact checkpoint reproducibility requires local training first)

Architecture 02 latent dimension: 70
Architecture 03 latent dimension: 70

----------------------------------------
CORRELATION RESULTS
----------------------------------------

Arch02:
ATT:
max: 0.854
mean: 0.410
median: 0.432

CBF:
max: 0.917
mean: 0.464
median: 0.480

Arch03:
ATT:
max: 0.850
mean: 0.468
median: 0.501

CBF:
max: 0.826
mean: 0.355
median: 0.343

----------------------------------------
LINEAR PROBE
----------------------------------------

                ATT R²     CBF R²
Arch02          0.858      0.989
Arch03          0.856      0.988

                ATT RMSE  CBF RMSE
Arch02          0.274      2.974
Arch03          0.276      3.106

----------------------------------------
REGIME ANALYSIS
----------------------------------------

0.5–1.0:
Arch02: Probe RMSE CBF = 4.083, Probe RMSE ATT = 0.418
Arch03: Probe RMSE CBF = 4.301, Probe RMSE ATT = 0.414

1.0–1.5:
Arch02: Probe RMSE CBF = 3.113, Probe RMSE ATT = 0.243
Arch03: Probe RMSE CBF = 3.016, Probe RMSE ATT = 0.243

1.5–2.0:
Arch02: Probe RMSE CBF = 2.504, Probe RMSE ATT = 0.253
Arch03: Probe RMSE CBF = 2.473, Probe RMSE ATT = 0.261

2.0–2.5:
Arch02: Probe RMSE CBF = 1.860, Probe RMSE ATT = 0.135
Arch03: Probe RMSE CBF = 1.806, Probe RMSE ATT = 0.141

2.5–3.0:
Arch02: Probe RMSE CBF = 2.832, Probe RMSE ATT = 0.245
Arch03: Probe RMSE CBF = 3.323, Probe RMSE ATT = 0.251

----------------------------------------
INTERPRETATION
----------------------------------------

Does Arch03 become more ATT-specific?
YES. The latent dimensions of Architecture 03 show a distinct shift: mean and median correlations with ATT increased, while mean, median, and max correlations with CBF significantly dropped. The representation is mathematically less CBF-sensitive.

Does latent analysis explain the short-ATT improvement?
PARTIALLY. The linear probe for CBF on Arch03 is actually worse in the 0.5-1.0s regime (RMSE 4.30 vs 4.08), meaning the latent space linearly contains less CBF information. However, the probe for ATT slightly improved (0.414 vs 0.418). This implies the short-ATT network improvement is not caused by z_kin becoming a "better linear CBF feature," but rather that a purer timing prior allows the non-linear CBF head to break shape/scale ambiguities more effectively without amplitude-gradient pollution.

Does latent analysis explain the overall degradation?
YES. By isolating the gradient, z_kin lost generalized CBF-sensitive features (overall probe CBF RMSE rose from 2.97 to 3.11). Because the CBF head concatenates this representation, starving z_kin of amplitude-coupled features harms predictions in the broader regimes where that joint optimization previously provided a mutually beneficial regularized feature space.

----------------------------------------
FINAL STATUS
----------------------------------------
PASS
