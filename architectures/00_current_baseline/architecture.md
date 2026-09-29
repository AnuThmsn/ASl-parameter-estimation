# Current Baseline Architecture

## Model Description
The current DNN is two independent standardized MLPs predicting CBF and ATT:
- **CBFnet:** 9 hidden layers x 50 neurons, ELU activations.
- **ATTnet:** 9 hidden layers x 100 neurons, ELU activations.
- **Loss:** MAE (L1Loss)

The forward model uses 4 PLDs: `[1.525, 2.025, 2.525, 3.025]`.
Noise model is dual-Rician magnitude: `X = sqrt((S+e1)^2+e2^2) + sqrt((S+e3)^2+e4^2)`.
