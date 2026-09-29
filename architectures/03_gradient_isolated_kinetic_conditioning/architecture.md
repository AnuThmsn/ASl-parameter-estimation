# Architecture 03: Gradient-Isolated Kinetic Conditioning

## Research Hypothesis
Architecture 02 demonstrated that feeding a learned kinetic representation (z_kin) into the CBF branch improves estimation in difficult regimes (like short ATT). However, because the CBF loss backpropagates through z_kin, the representation becomes a mixed feature set rather than a pure kinetic/timing prior.
**Hypothesis:** By preventing the CBF objective from backpropagating through z_kin (via gradient isolation), the kinetic representation will be strictly shaped by the ATT objective, leading to a cleaner timing prior that might further improve CBF conditioning.

## Difference from Architecture 02
The architecture is identical in structure and parameter count (~94,772) to Architecture 02. The ONLY difference is the explicit detach() applied to z_kin before concatenating it with the shared features h for the CBF branch.

`	ext
Input [S1 S2 S3 S4]
          |
    Shared Encoder
          |
          h
          |
   Kinetic Encoder
          |
        z_kin
       /      \
      |        |
 ATT Head   STOP GRADIENT
               |
         detach(z_kin)
               |
       concat(h, z_kin_detached)
               |
          CBF Encoder
               |
          CBF Head
`

## Gradient Flow
*   **ATT objective:** Backpropagates through tt_head, kinetic_encoder, and shared_encoder.
*   **CBF objective:** Backpropagates through cbf_head, cbf_encoder, and shared_encoder. It does **not** flow into kinetic_encoder.

## Expected Outcomes & Failure Modes
*   **Outcome A:** Performance improves, proving a strict timing prior is optimal.
*   **Outcome B:** Performance degrades, proving that joint optimization of the latent representation by both objectives is mutually beneficial for generalized estimation.
