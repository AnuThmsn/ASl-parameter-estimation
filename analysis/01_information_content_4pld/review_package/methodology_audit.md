# Methodology Audit: Information-Content Analysis

## 1. Current Method
The previous information-content analysis was based on a dense grid evaluated over the physiological ranges of CBF [10, 100] and ATT [0.5, 3.0]. Derivatives (the Jacobian) were calculated using a one-sided finite-difference approximation. Singular values and condition numbers were computed using standard SVD. Ambiguity was defined as parameter pairs whose signal vector Euclidean distance was smaller than the SNR=10 noise standard deviation, despite being well-separated in parameter space.

## 2. Mathematical Definition
- **Finite-Difference Jacobian (Previous):** 
  $J_{ij} = \frac{S_i(\theta_j + h_j) - S_i(\theta_j)}{h_j}$
  where $h_{CBF} = 10^{-3}$ ml/100g/min and $h_{ATT} = 10^{-3}$ s.
- **Parameter Scaling:** Parameters were scaled using approximate Z-scores:
  $z_{CBF} = (CBF - 50) / (100/\sqrt{12})$ and $z_{ATT} = (ATT - 1.75) / (2.5/\sqrt{12})$.
- **SVD & Condition Number:** $J = U \Sigma V^T$. Condition number $\kappa = \sigma_1 / \sigma_2$.
- **Column Correlation:** $\text{corr}(J_{CBF}, J_{ATT}) = \frac{J_{CBF} \cdot J_{ATT}}{\|J_{CBF}\| \|J_{ATT}\|}$.
- **Ambiguity Distance:** $d_{sig} = \|S(\theta_a) - S(\theta_b)\|$, $d_{param} = \sqrt{(\Delta z_{CBF})^2 + (\Delta z_{ATT})^2}$.
- **Fisher Information:** $F = J^T \Sigma^{-1} J$, where $\Sigma = \text{diag}(\sigma_{noise}^2)$.

## 3. Potential Numerical Problems
1. **Finite-Difference Step Size:** A step size of $10^{-3}$ for CBF (which ranges up to 100) is a relative step of $10^{-5}$, which is reasonably safe. For ATT, $10^{-3}$ is a relative step of $10^{-3}$, which might encounter floating-point truncation issues or hit the non-differentiable boundaries of the piecewise signal model.
2. **Piecewise Continuity:** The true signal model contains $\max(PLD - ATT, 0)$. This function is continuous, but its derivative is a step function (discontinuous). Finite differences straddling these boundaries will yield inaccurate gradients.
3. **Condition Number Inflation:** When $S \approx 0$ (e.g., long ATT where no bolus has arrived), $J \approx 0$. Numerical noise causes $\sigma_2$ to approach machine epsilon, falsely inflating the condition number to infinity rather than just a very poorly conditioned problem.
4. **Fisher Information Invertibility:** In regions where condition number is extremely high, $F$ becomes singular or numerically non-positive definite, leading to negative variances if inverted naively.

## 4. Potential Interpretation Problems
1. **Ambiguity vs. Density:** The ambiguity search originally used a coarse sampling, which might miss the worst-case continuous ambiguity manifolds (valleys of identical signal).
2. **Correlation vs Identifiability:** A high column correlation (near 1.0) indicates that infinitesimal changes in CBF and ATT produce collinear signal changes. However, because the system is non-linear, macroscopic changes might still curve apart.

## 5. Proposed Validation Tests
- **Experiment A:** Sweep finite difference step sizes $h_{CBF} \in \{0.1, 0.5, 1.0, 2.0\}$ and $h_{ATT} \in \{0.001, 0.005, 0.01, 0.02, 0.05\}$ to verify stability.
- **Experiment B:** Derive and implement the exact analytical Jacobian to serve as ground truth for the finite difference approximation, explicitly handling the piecewise boundaries.
