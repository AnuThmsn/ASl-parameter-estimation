# Architecture 02: Hierarchical Kinetic Conditioning

## Research Hypothesis
ATT primarily determines the kinetic/timing regime of the ASL signal. CBF estimation can then benefit from conditioning on the learned kinetic representation.

Instead of simply sharing features between CBF and ATT as latent siblings, explicitly make the representation hierarchical: extracting a kinetic representation first to infer timing structure, then providing that timing structure (along with general features) to the CBF branch.

## Motivation from ASL Inverse Problem
The information-content analysis established that:
1. CBF acts as a uniform amplitude/scale-related parameter.
2. ATT changes the kinetic/timing structure of the signal and acts conditionally depending on the regime.
3. Short ATT and long ATT regions present structural non-identifiability where CBF and ATT can mimic each other (strong sensitivity collinearity).
By isolating the kinetic structure (ATT) into its own latent representation and conditionally feeding it into the scale-extraction branch (CBF), the network may better disambiguate scale from shape changes compared to symmetric independent processing.

## Difference between Architecture 00, 01, and 02
*   **Architecture 00**: Completely independent networks for CBF and ATT. No shared information.
*   **Architecture 01**: A generic shared encoder creates a common representation h, which branches independently into CBF and ATT heads.
*   **Architecture 02**: Extracts h, then extracts a dedicated kinetic latent representation z_kin from h. z_kin predicts ATT. The CBF branch is computed conditionally by receiving concat(h, z_kin). 

## Complete Architecture
`	ext
           [S1 S2 S3 S4]
                 │
        input normalization
                 │
  shared encoder (4 → 90, 7 layers)      ← h
                 │
kinetic encoder (90 → 70, 2 layers)      ← z_kin
      ┌──────────┴──────────┐
      │                     │
   ATT head         concat(h, z_kin) → 160 dims
 (70 → 40 → 1)              │
      │              cbf encoder (160 → 90, 3 layers)
   ATT_hat                  │
                     cbf head (90 → 1)
                            │
                         CBF_hat
`

## Layer-by-layer Dimensions
*   **Shared Encoder**: Linear(4→90), ELU, [Linear(90→90), ELU] x 6. Output: 90.
*   **Kinetic Encoder**: Linear(90→70), ELU, Linear(70→70), ELU. Output: 70.
*   **ATT Head**: Linear(70→40), ELU, Linear(40→1). Output: 1 (ATT).
*   **CBF Encoder**: Concat(90, 70) → 160 dims. Linear(160→90), ELU, [Linear(90→90), ELU] x 2. Output: 90 dims.
*   **CBF Head**: Linear(90→1). Output: 1 (CBF).
*   **Total Trainable Parameters**: ~ 94,772

## How Kinetic Representation is Produced
The model processes the shared generic latent features through an explicitly dedicated multi-layer network branch (kinetic_encoder) yielding z_kin. 

## How CBF is Conditioned
The CBF encoder does not just receive the shared features; it receives a concatenation of h and z_kin. This allows the CBF extraction logic to adjust its scale-reading behavior based on the learned timing regime.

## Why Predicted ATT is Not Directly Fed into CBF
Directly feeding the single scalar output ATT_hat into the CBF branch creates direct error propagation (if ATT_hat is noisy, CBF will explicitly shift in response). Additionally, z_kin contains a higher-dimensional distributed representation of the kinetic structure rather than just a compressed scalar, preserving richer timing context for the CBF branch.

## Expected Advantage
If the hypothesis is correct, Architecture 02 should exhibit lower CBF RMSE, specifically in ambiguous kinetic regimes (like very short ATT), compared to Architecture 01 and the parameter-matched ablation.

## Potential Failure Modes
*   The network may simply ignore z_kin when extracting CBF, devolving into the equivalent of Architecture 01.
*   The hierarchical dependence might cause instability or premature convergence compared to fully independent models.

## What Result Would Falsify the Hypothesis
If Architecture 02 performs equivalently (within seed variability) to the HierarchicalKineticNet_NoCondition ablation, we must conclude that hierarchical conditioning of CBF on kinetic latents provides no measurable structural advantage for this 4-PLD ASL problem.
