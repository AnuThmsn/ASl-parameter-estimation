# Architecture Documentation

1. Literature motivation
U-Net architectures are well-suited for medical imaging spatial contexts.

2. Original Luciw et al. architecture
3D U-Net for ASL.

3. Differences between paper and our implementation
4 PLDs instead of 6, channel-wise mean/std instead of 95th percentile.

4. Input representation (4 channels x D x H x W)
4 PLDs as input.

5. Encoder (3 levels, stride=(2,2,1) downsampling)
Three levels with stride (2,2,1).

6. Bottleneck
Standard 3D convolutions.

7. Decoder (3 levels, stride=(2,2,1) upsampling)
Matching the encoder.

8. Skip connections
Concatenated features.

9. Output heads (1x1x1 conv -> 2 channels)
Outputting CBF and ATT.

10. Loss (brain MAE + background MAE)
L1 loss masked.

11. Normalization
Channel-wise mean and std.

12. Parameter count (to be filled after running model)
TBD

13. Why U-Net is relevant to ASL parameter estimation
Spatial coherence helps regularize.

14. Why U-Net may help reduced-PLD estimation
Spatial context fills in missing temporal samples.

15. Limitations
Blurring of sharp boundaries.
