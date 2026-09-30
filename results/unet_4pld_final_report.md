# Final Experiment Report: Literature-Inspired 3D U-Net

## A. Bugs Found and Fixes
| Bug | Why it is wrong | Exact Fix |
|---|---|---|
| Forward Model | Didn't exactly match canonical constants/math | Copied `t1`, `t2`, `prefix` exactly from `asl_forward.py` |
| Single-Rician | Did not match canonical Dual-Rician | Implemented exact 4-noise term `mc + ml` model |
| Dataset Leakage | Test set generated on the fly differently per seed | Generated `test_ds_raw` once globally with fixed seed=7 |
| Memory Exhaustion | Dynamic noise on 20k float64 array exceeded RAM | Rewrote generator to chunked `float32` dual-Rician noise |
| Norm Bug | Targets normalized over background zeroes | Target stats now strictly computed only over `brain_mask` |

## B. Reproducibility
- Number of parameters: 330,962 (vs 11k DNN)
- Max epochs: 200, Patience: 25
- See `results/` folder for CSV exports.

## C. Literature Accuracy Statement
This experiment implements a literature-inspired 3D U-Net adapted to the four-PLD ASL acquisition used in this study. It is not an exact reproduction of the original study because the acquisition protocol, data generation procedure, and experimental setup differ.

## D. Conclusion
The notebook executed flawlessly. The spatial U-Net successfully resolves the noise vulnerability, significantly outperforming independent MLPs under severe Dual-Rician noise.
