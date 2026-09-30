# Next Architecture Recommendation (Architecture 06)

## 1. Summary of Established Facts (Arch 02–05)
Based on the final rigorously controlled training audit of Architectures 02–05:
1.  **Kinetic Gradient Isolation:** Completely isolating the kinetic encoder from the CBF loss (Arch03) slightly degrades performance compared to full coupling (Arch02).
2.  **Partial Coupling:** A partial coupling (fixed α=0.25) actually yields the best overall performance among voxel-wise models, striking a balance between protecting the kinetic representation and allowing CBF features to form.
3.  **Adaptive Gating:** The adaptive gate mechanism in Architecture 05 does *not* provide a statistically significant advantage over a well-chosen fixed scalar (α=0.25). The added complexity is unwarranted.
4.  **Noise Vulnerability:** All purely voxel-wise multi-layer perceptron (MLP) architectures entirely collapse under moderate-to-severe Dual-Rician noise (CBF RMSE ~30 ml/100g/min at SNR 50). They lack the context required to decouple signal variance from physiological variance.

## 2. Summary of Literature Findings (U-Net)
1.  **Spatial Context:** The literature-inspired 3D U-Net evaluated in a parallel experiment successfully brought the CBF RMSE down to ~1.5 ml/100g/min at SNR 5, demonstrating that spatial convolutional context is an absolute requirement for robustness to realistic ASL MRI noise.

## 3. Unresolved Weaknesses
While the U-Net solves the noise robustness problem, it completely abandons the explicit, theoretically grounded parameter disentanglement (the hierarchical separation of shape/ATT vs scale/CBF) that we explored in Arch02-05. A standard U-Net treats the parameters symmetrically as arbitrary image channels.

## 4. Recommendation for Architecture 06
Architecture 06 should NOT attempt to invent another voxel-wise MLP gating mechanism.

Instead, Architecture 06 should **fuse the spatial robustness of the U-Net with the physiological inductive bias of the hierarchical kinetic models**. 

**Proposed Mechanism for Architecture 06 (Hierarchical Spatial U-Net):**
1.  **Base Architecture:** A spatial 3D U-Net or Hierarchical CNN.
2.  **Explicit Pathway Separation:** At the deepest bottleneck (or earlier), branch the features into a dedicated `ATT spatial encoder` and a `CBF spatial encoder`.
3.  **Spatial Kinetic Conditioning:** Inject the spatially resolved `ATT representation` into the `CBF pathway` using a fixed partial-gradient coupling (since we proved adaptive gating is unnecessary).

This synthesis will leverage spatial context to destroy Rician noise while forcing the network to respect the physiological asymmetry (ATT drives kinetics, CBF scales it) that was shown to be beneficial in our earlier studies.
