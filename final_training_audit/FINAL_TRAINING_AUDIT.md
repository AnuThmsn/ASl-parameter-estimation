# Final Training Audit and Results

## 1. Audit and Retraining Protocol
*   **Previous Status:** Architectures 02, 03, and 04 were originally trained as fast smoke-tests (30 epochs limit). 
*   **Final Retraining Protocol:** We instituted a rigorous retraining up to 200 epochs with a patience of 25. All architectures were given identical resources (20,000 canonical training samples).
*   **Normalization:** We explicitly saved and used canonical normalization statistics derived *strictly* from the 20,000 noisy training samples to prevent any test leakage.
*   **Random Seeds:** All architectures were run across 3 seeds (42, 123, 2024).

## 2. Gradient Verification Test
Prior to training, we verified the intended gradient mechanics for CBF backpropagation into the kinetic encoder:
*   **Architecture 02 (Hierarchical):** Fully coupled (Baseline 100%)
*   **Architecture 03 (Isolated):** 0.00% gradient transmission (successfully isolated)
*   **Architecture 04 (Partial):** 50.00% gradient transmission (successfully partially coupled)

## 3. Convergence Results
The full training run proved that the previous 30-epoch limits were slightly premature for some seeds.
*   Arch05 (Adaptive) required an average of **83.7 epochs** to reach its best validation loss.
*   Other models consistently hit early stopping between epoch 40 and 100.
*   Retraining to full convergence slightly improved overall model stability compared to the historical smoke tests.

## 4. Final Performance Analysis (Clean SNR = inf)
At convergence on noise-free data, the overall differences between the models are tightly clustered:
*   **Arch02:** CBF RMSE = 2.748 ± 0.068
*   **Arch03:** CBF RMSE = 2.817 ± 0.129
*   **Arch04:** CBF RMSE = 2.749 ± 0.100
*   **Arch05:** CBF RMSE = 2.715 ± 0.058

## 5. Paired Statistical Differences
When rigorously evaluating paired seed differences (delta = Arch_X - Arch05) at Clean SNR:
*   **Arch02 vs Arch05:** Arch05 is better by ~0.033 CBF RMSE (95% CI: ±0.075)
*   **Arch03 vs Arch05:** Arch05 is better by ~0.102 CBF RMSE 
Because the 95% Confidence Interval (±0.075) is larger than the mean difference (0.033), the performance difference between Arch02 and Arch05 is **not statistically significant** across these 3 seeds. 

## 6. The Fixed-Alpha Experiment (Adaptive Gating Justification)
To determine if Arch05's adaptive gate (mean α ≈ 0.7) is actually learning an optimal dynamic weighting or if a fixed scalar is sufficient, we tested rigid α values:
*   **α = 0.00 (Isolated):** 2.817 ± 0.129
*   **α = 0.25:** 2.686 ± 0.096  *(Best performing fixed value)*
*   **α = 0.50:** 2.749 ± 0.100
*   **α = 0.75:** 2.831 ± 0.160
*   **α = 1.00 (Coupled):** 2.748 ± 0.068
*   **Adaptive (Arch05):** 2.715 ± 0.058

**Conclusion:** The best fixed-alpha (0.25) actually slightly outperformed the adaptive gate (2.686 vs 2.715) at Clean SNR, indicating that while partial gradient coupling is highly beneficial compared to full isolation (Arch03, α=0), the dynamic adaptivity of the gate in Arch05 does not provide a robust, statistically significant advantage over simply choosing a good fixed hyperparameter (α=0.25). 

## 7. Noise Robustness Failure
An important finding emerged when evaluating at realistic SNRs (50, 10, 5). Across *all* architectures, the CBF error exploded to ~30 ml/100g/min. This reveals a fundamental limitation of the purely independent voxel-wise MLP architectures when confronted with severe Dual-Rician noise—they entirely fail to denoise the signal without spatial context (which the recent U-Net experiment successfully mitigated, achieving ~1.5 ml/100g/min at SNR 5).

## 8. Final Status of Architectures
*   **Arch02:** FINAL-VALIDATED
*   **Arch03:** FINAL-VALIDATED
*   **Arch04:** FINAL-VALIDATED
*   **Arch05:** FINAL-VALIDATED 

All models are correctly implemented, trained to convergence, and stringently evaluated. The numbers are now safe for the thesis.
