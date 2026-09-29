# Derivative Validation (Experiment B)

## Analytical Derivative Derivation

The ASL forward model for a single PLD is:

$$ S = K \cdot CBF \cdot \exp(-ATT/T_{1a}) \cdot \left( \exp\left(-\frac{\max(PLD - ATT, 0)}{T_{1t}}\right) - \exp\left(-\frac{\max(PLD + \tau - ATT, 0)}{T_{1t}}\right) \right) $$

where $K = \frac{2 \alpha \beta T_{1t}}{6000 \lambda}$.

### Derivative with respect to CBF
The signal equation is strictly linear in CBF. Therefore, the analytical derivative is simply the signal divided by CBF:
$$ \frac{\partial S}{\partial CBF} = \frac{S}{CBF} $$

### Derivative with respect to ATT
The derivative with respect to ATT involves the chain rule on the piecewise terms. Let the exponential term be $E_{term}$.
$$ \frac{\partial S}{\partial ATT} = -\frac{1}{T_{1a}} S + K \cdot CBF \cdot \exp(-ATT/T_{1a}) \cdot \frac{\partial E_{term}}{\partial ATT} $$

The derivative of the piecewise terms $\max(PLD - ATT, 0)$ is:
- If $ATT < PLD$: $-1$
- If $ATT > PLD$: $0$

Thus, the analytical derivative exists and is well-defined almost everywhere (except exactly at $ATT = PLD$ and $ATT = PLD + \tau$). We implemented this explicitly in `validation_experiments.py`.

## Validation Results

We compared the analytical Jacobian against the finite-difference Jacobian across multiple step sizes (Experiment A).

For a representative point (CBF=50, ATT=1.6):
- The relative error for CBF sensitivity ($\frac{\|J_{num} - J_{ana}\|}{\|J_{ana}\|}$) drops to near zero ($10^{-14}$) regardless of step size, because the function is perfectly linear in CBF. Any finite-difference step produces the exact analytical derivative.
- The relative error for ATT sensitivity scales linearly with the finite-difference step size $h_{ATT}$. For $h_{ATT} = 0.001$, the relative error is on the order of $10^{-4}$, which confirms the analytical derivation is correct and the numerical approximation was reasonably accurate, provided it did not step across a PLD boundary.

### Collinearity at Short ATT
The finite-difference table and analytical solutions show that for ATT values below the first PLD (e.g., 0.5–1.5s), the standardized Jacobian columns become essentially collinear, with cosine similarity $\approx 1$ and condition numbers in the $10^{12} - 10^{14}$ range. 
When $ATT < PLD_i$, both exponential terms are in the same branch, so the PLD-dependent part factors out. The signal behaves like:
$S_i \approx CBF \cdot A(ATT) \cdot B(PLD_i)$
Because of this, changing CBF and changing ATT both act primarily as scale changes in this regime, leading to severe collinearity.
