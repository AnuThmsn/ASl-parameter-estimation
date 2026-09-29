# Architecture 01: Shared Kinetic Representation

This folder contains the implementation, training, and evaluation code for Architecture 01, which tests the hypothesis that a shared feature encoder improves CBF and ATT parameter estimation.

## Reproduction Commands

To reproduce the training and ablation studies (which will run 3 seeds across 3 architectures: Baseline, Architecture01, AblationIndependent):
```bash
python architectures/01_shared_kinetic_representation/train.py
```
*Note: This acts as a smoke test using a subset of the data (20k training samples) to ensure it can run efficiently.*

To evaluate the trained Architecture 01 (Seed 42) and generate diagnostic plots and error distributions:
```bash
python architectures/01_shared_kinetic_representation/evaluate.py
```

## Expected Outputs

- **Checkpoints:** Stored in `checkpoints/` as `.pt` files.
- **Results Table:** The aggregated multi-seed SNR comparisons are saved to `comparisons/architecture_00_vs_01.csv`.
- **Diagnostic Tables:** Error by CBF/ATT range is saved in `tables/error_by_att_range.csv` and `tables/error_by_cbf_range.csv`.
- **Diagnostic Figures:** Stored in `figures/`, including CBF monotonicity, ATT response, and Shared Latent PCA plots.
