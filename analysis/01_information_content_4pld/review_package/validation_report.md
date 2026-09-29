# Final Scientific Validation Report

## Validation Questions & Answers

1. **Is the numerical Jacobian stable?**
   Yes, provided the step size is small ($h < 0.01$). However, the analytical Jacobian is perfectly stable and was derived and utilized for this validation phase.
2. **Are the sensitivity results reproducible across finite-difference steps?**
   Yes. For CBF, because the model is perfectly linear, the sensitivity is identical regardless of the step size. For ATT, the sensitivity converges linearly with step size.
3. **Are the results consistent with analytical derivatives?**
   Yes, the finite-difference approach matched the analytical derivatives with relative errors below $10^{-4}$ for small step sizes.
4. **What does changing CBF do to the four measurements?**
   CBF acts as a pure, uniform, linear scaling factor across all four PLDs simultaneously. $S \propto CBF$.
5. **What does changing ATT do?**
   Changing ATT shifts the arrival time of the kinetic curve, changing which PLDs observe zero signal versus non-zero signal, and altering the apparent $T_1$ decay envelope.
6. **Which PLDs provide complementary information?**
   Early PLDs (1.525, 2.025) dictate the bolus arrival and distinguish short ATTs. Late PLDs (2.525, 3.025) capture the decay tail, providing necessary information to untangle the CBF scale from the T1 decay rate for long ATTs.
7. **Why do the sensitivity curves change around the PLDs?**
   The forward model contains piecewise terms: $\max(PLD - ATT, 0)$. When ATT crosses a PLD boundary, a signal term either switches on or switches off, creating a discontinuity in the sensitivity (the derivative).
8. **Where is the inverse problem well-conditioned?**
   It is well-conditioned at short ATTs ($ATT < 1.5s$), where multiple PLDs capture the non-zero kinetic curve.
9. **Where is it ill-conditioned?**
   It is severely ill-conditioned at long ATTs ($ATT > 2.5s$). Here, the early PLDs contain zero signal, leaving only 1 or 2 PLDs to determine two independent parameters.
10. **Where are CBF and ATT locally difficult to separate?**
    At long ATTs. The Jacobian columns become nearly parallel (cosine similarity $\approx 1.0$). A small increase in CBF (scaling up) can be perfectly masked by a small increase in ATT (shifting later and decaying more), producing identical observations.
11. **How does noise affect identifiability?**
    Noise destroys the fine shape differences in the tail of the kinetic curve. Fisher Information matrices at SNR 10 show that the variance of the ATT estimate explodes at long ATTs.
12. **At what SNR does ambiguity become significant?**
    Ambiguity is severe even at standard operating SNRs (SNR 10). Signal distances between distant parameter pairs can be smaller than 1 standard deviation of noise.
13. **What parameter combinations are ambiguous?**
    Pairs where one has higher CBF and shorter ATT, and the other has lower CBF and longer ATT. For example, `CBF:66.8, ATT:1.33` vs `CBF:81.1, ATT:0.50` produce visually identical signal vectors within the noise envelope.
14. **What information is genuinely recoverable from four PLDs?**
    For short to medium ATTs, both scale and shape are recoverable.
15. **What information cannot reliably be recovered?**
    For long ATTs, uniquely resolving both CBF and ATT is fundamentally impossible from only 4 PLDs without strong prior assumptions or regularization.

## Implications for Neural Network Design

**OBSERVATION:** CBF acts as a pure linear scalar on the signal, while ATT non-linearly shifts the signal. At long ATTs, the problem is ill-conditioned and ambiguous.
**INTERPRETATION:** A network cannot rely solely on localized linear features to untangle CBF and ATT. It must learn the holistic shape of the kinetic curve. Because identical signal vectors can arise from different true parameters, independent point-estimate networks will struggle or output noisy, unstable predictions in the ambiguous regions.
**ARCHITECTURAL HYPOTHESIS:** The evidence strongly supports **B. shared encoder + separate heads**, and specifically points toward **C. hierarchical representation**.
If the network first estimates a shared latent representation of the curve's "shape" (dominated by ATT), it can use this representation to condition the linear scaling estimate (CBF). An architecture that explicitly shares the early feature-extraction layers forces the network to learn a single physically consistent kinetic curve representation, mitigating the extreme ill-conditioning seen at long ATTs.
