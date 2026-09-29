# ASL Parameter Estimation: Project Audit (Architectures 00-04)
**Date:** 2026-09-29

## Overview
This project studies whether Deep Neural Networks (DNNs) can recover ASL MRI parameters (CBF, ATT) from an undersampled 4-PLD protocol. The core focus has been the topological relationship and gradient flow between the timing/shape parameter (ATT) and the scale/amplitude parameter (CBF).

## Architectures Implemented & Evaluated
1. **Architecture 00 (Current Baseline)**
   * **Structure:** Independent DNNs for CBF and ATT.
   * **Result:** Serves as the disconnected baseline. Subject to high error in ambiguous boundary regimes.

2. **Architecture 01 (Shared Representation)**
   * **Structure:** Shared trunk mapping PLDs to a latent representation, branching into independent CBF and ATT heads.
   * **Result:** Evaluated via Information Content Analysis.

3. **Architecture 02 (Hierarchical Kinetic Conditioning)**
   * **Structure:** z_kin (kinetic representation) is learned and concatenated directly into the CBF branch. 100% of the CBF loss backpropagates through z_kin.
   * **Result:** Improved general performance, but latent representation was "polluted" with amplitude features, failing to resolve the hardest shape/scale ambiguities.

4. **Architecture 03 (Gradient-Isolated Kinetic Conditioning)**
   * **Structure:** Same as Arch 02, but z_kin is detached before entering the CBF branch. 0% of CBF loss backpropagates to the kinetic encoder.
   * **Result:** Overall CBF RMSE degraded compared to Arch 04. However, it achieved the **best performance** in the most ambiguous short-ATT regime (0.5-1.0s), proving that a strict, unpolluted timing prior is strictly necessary for boundary limits.

5. **Architecture 04 (Controlled Partial Gradient Kinetic Conditioning)**
   * **Structure:** lpha = 0.5 scaling on the backward pass from the CBF branch to the kinetic encoder. 50% of CBF loss backpropagates.
   * **Result:** Achieved the **best overall performance** across standard regimes (lowest mean CBF RMSE at SNR > 10). However, it completely collapsed in the short-ATT (0.5-1.0s) regime, validating the hypothesis that any amplitude gradient mixing destroys the pure kinetic prior needed for boundaries.

## Scientific Conclusion So Far
We have successfully isolated the fundamental mechanistic trade-off in multi-parameter ASL estimation:
* **Joint Optimization (50-100% Coupling):** Acts as a powerful feature regularizer, yielding superior overall predictions in standard physiological ranges.
* **Strict Disentanglement (0% Coupling):** Necessary to resolve extreme mathematical collinearity (short ATT / amplitude-timing ambiguities). 
No uniform gradient routing architecture optimally solves both regimes simultaneously.
