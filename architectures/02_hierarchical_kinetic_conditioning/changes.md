# Change Log - Architecture 02

## v0.1.0
*   Implemented HierarchicalKineticNet and HierarchicalKineticNet_NoCondition based on the Hierarchical Kinetic Conditioning hypothesis.
*   Total parameter count tuned to ~ 94.7k (Full) and ~ 106.8k (No-Condition ablation), matching the capacity constraints of Architecture 00 (102k) and Architecture 01 (91k).
*   Corrected SNR noise evaluation pipeline explicitly imported from Architecture 01's audit to strictly use add_noise(S_true, sd).
*   Regime-specific error analysis generated for ATT bins (0.5-3.0s) and CBF bins (low, medium, high).
