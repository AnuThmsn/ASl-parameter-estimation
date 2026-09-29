# Final Scientific Validation Report

## Validation Questions & Answers

1. **Is the numerical Jacobian stable?**
   Yes, provided the step size is small. However, the analytical Jacobian is perfectly stable and was derived and utilized for this final validation phase.
2. **Are the results consistent with analytical derivatives?**
   Yes, the finite-difference approach matched the analytical derivatives with relative errors below $10^{-4}$ for small step sizes.
3. **What does changing CBF do to the four measurements?**
   CBF acts as a pure, uniform, linear scaling factor across all four PLDs simultaneously. $S \propto CBF$.
4. **What does changing ATT do?**
   ATT is scale-like in some kinetic regimes (especially before the first PLD) and becomes increasingly shape-discriminative when the ATT crosses the measurement PLDs. 
5. **Why do the sensitivity curves change around the PLDs?**
   Once ATT crosses a PLD, that PLD moves into another piece of the kinetic model. The response is no longer simply a common scaling. The shape across measurements changes:
   - `ATT < PLD1`: mostly scale-like ambiguity, CBF and ATT are hard to separate.
   - `ATT crosses PLD1`: one measurement changes regime, shape information appears.
   - `ATT crosses PLD2/3`: other measurements change regime, providing distinct information.
6. **Where is the inverse problem well-conditioned?**
   It is well-conditioned strictly in intermediate ATTs where the curve is straddling the measurement PLDs, allowing shape to be heavily discriminated from pure scale.
7. **Where is it ill-conditioned?**
   It is severely ill-conditioned at short ATTs ($ATT < 1.5s$) because the kinetic curves are on the same branch and act dominantly as scale factors. It is also ill-conditioned at very long ATTs where bolus arrival is delayed and early PLDs contain pure noise.
8. **Where are CBF and ATT locally difficult to separate?**
   At short ATTs ($ATT < PLD_1$). The Jacobian columns become nearly parallel (cosine similarity $\approx 1.0$). A small increase in CBF can be perfectly masked by an equivalent structural shift in ATT.
9. **How does noise affect identifiability?**
   Fisher Information matrices (calculated empirically from the exact Rician-sum Monte Carlo simulations) show that the parameter uncertainty bounds explode in the collinear regions. 
10. **At what SNR does ambiguity become significant?**
    Ambiguity is a smooth distribution depending on SNR. At SNR 50, ambiguity is minor (117 distant pairs confuse the model). At SNR 10, ambiguity becomes a severe structural issue (769 distant pairs confuse the model).
11. **What information cannot reliably be recovered?**
    For extreme short and extreme long ATTs, uniquely resolving both CBF and ATT is fundamentally impossible from only 4 PLDs because they project onto the identical scale-like direction in signal space.

## Implications for Neural Network Design

**OBSERVATION:** CBF acts as a pure linear scalar on the signal, while ATT acts as a scale shift in early regimes and a shape modifier at PLD boundaries. 
**INTERPRETATION:** A generic "shared multitask network" is insufficient because it treats the two parameters as identical latent siblings. The information structure is highly specific: the measurements contain scale information and kinetic/regime information separately.
**ARCHITECTURAL HYPOTHESIS (Architecture 01):**
Our network should explicitly distinguish these two latent quantities. A highly promising candidate is:
```text
           [S1 S2 S3 S4]
                 │
        input normalization
                 │
           shared encoder
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
kinetic/regime       amplitude/scale 
representation       representation
      │                     │
      ▼                     │
     ATT                    │
      │                     │
      └─────────┬───────────┘
                ▼
          CBF estimation
```
This physically motivated hierarchy explicitly models the forward process where the kinetic regime (ATT) dictates the shape, and the amplitude acts as a terminal scale (CBF).
