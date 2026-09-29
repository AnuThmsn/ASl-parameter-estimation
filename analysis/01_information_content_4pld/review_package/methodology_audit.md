# Methodology Audit: Information-Content Analysis

## 1. Current Method
The revised information-content analysis was based on a dense grid evaluated over the physiological ranges of CBF [10, 100] and ATT [0.5, 3.0]. Derivatives (the Jacobian) were calculated analytically using the exact piece-wise forward equations. Singular values and condition numbers were computed using standard SVD. Ambiguity was evaluated by exploring the distribution of signal distances and parameter distances, thresholding against multiple noise levels (SNR=5, 10, 20, 50). Fisher information was generated using an empirical covariance matrix drawn from the exact Rician-sum noise simulator.

## 2. Mathematical Definition
- **Analytical Jacobian:** Derived exactly. $\frac{\partial S}{\partial CBF} = \frac{S}{CBF}$ (perfectly linear). The derivative with respect to ATT is computed using the chain rule on the continuous piecewise elements $\max(PLD - ATT, 0)$.
- **Parameter Scaling:** Parameters were scaled using approximate Z-scores:
  $z_{CBF} = (CBF - 50) / (100/\sqrt{12})$ and $z_{ATT} = (ATT - 1.75) / (2.5/\sqrt{12})$.
- **SVD & Condition Number:** $J = U \Sigma V^T$. Condition number $\kappa = \sigma_1 / \sigma_2$.
- **Column Correlation:** $\text{corr}(J_{CBF}, J_{ATT}) = \frac{J_{CBF} \cdot J_{ATT}}{\|J_{CBF}\| \|J_{ATT}\|}$.
- **Fisher Information:** $F = J^T \Sigma^{-1} J$, where $\Sigma$ is empirically estimated by Monte Carlo sampling from the dual-Rician magnitude operation.

## 3. Rectified Problems
1. **Ambiguity Analysis:** The previous arbitrary threshold was discarded. Ambiguity is now presented as a continuous distribution of distant parameter pairs that generate identical signals within noise envelopes at various SNRs.
2. **Fisher Information:** The previous diagonal Gaussian noise assumption was corrected. The actual observation model $X = \sqrt{(S+e_1)^2+e_2^2} + \sqrt{(S+e_3)^2+e_4^2}$ generates non-trivial covariance structures, especially at low signal levels. The Fisher matrix now uses empirically derived 4D covariance matrices from the exact forward simulator.
3. **Condition Number Interpretation:** High condition numbers at short ATTs (< PLD1) correctly reflect the fact that all measurements fall on the identical piece of the kinetic function, making both CBF and ATT act strictly as scale factors, resulting in extreme collinearity. This invalidates previous assumptions that early ATTs were universally well-conditioned.
