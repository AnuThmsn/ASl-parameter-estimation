# Audit: Existing Repository State

1. **How Arch02-05 were trained**:
   - Trained on 20,000 generated ASL samples, validated on 5,000, 30 epochs max.
   - Used Adam optimizer (1e-3), L1 loss, batch size 512, gradient clip 1.0.
   - Normalization: noisy canonical normalizer (
_noise_levels=100, sd_max=66.84).
   - Checkpoints save the *best validation* weights.

2. **Checkpoints**:
   - Arch02: seeds 42, 123, 2024 exist.
   - Arch03: seeds 42, 123, 2024 exist.
   - Arch04: seeds 42, 123, 2024 exist.
   - Arch05: seeds 42, 123, 2024 exist (seed 2024 was generated just before this study).
   - All correspond exactly to the reported experiments.

3. **Normalization and Data Generation**:
   - All models used the exact same 
_noise_levels=100, sd_max=334.2039/5.0 normalization scheme.
   - All models used the exact identical generate_data configurations for training.

4. **Safety**:
   - I will *not* overwrite rchitectures/ checkpoints. 
   - I will train generalized fixed-alpha models explicitly inside experiments/gradient_coupling_study/checkpoints for 100% control over the comparison sweep.
