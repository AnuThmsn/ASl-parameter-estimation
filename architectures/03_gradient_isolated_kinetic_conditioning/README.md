# Gradient-Isolated Kinetic Conditioning (Architecture 03)

This directory contains the implementation of Architecture 03, which explicitly detaches the kinetic latent representation (z_kin) before it is used to condition the CBF prediction. 

The goal of this architecture is to strictly prevent CBF amplitude loss from backpropagating into the timing/shape features extracted by the kinetic encoder, allowing us to study the properties of a pure kinetic prior.

*   **Implementation:** GradientIsolatedKineticNet inside model.py.
*   **Results:** See esults.md.
