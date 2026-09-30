# U-Net Experiment for ASL Parameter Estimation

Paper: Luciw et al., "Automated generation of cerebral blood flow and arterial transit time maps from multiple delay arterial spin-labeled MRI" (2022)
Architecture: 3D U-Net with skip connections
Paper input: 6-PLD human in-vivo ASL spatial volumes
Our input: 4-PLD synthetic ASL spatial volumes [4, D, H, W]
Paper output: Voxel-wise CBF and ATT maps
Our output: Voxel-wise CBF and ATT maps [2, D, H, W]
Paper dataset: Real human 3D ASL scans
Our dataset: Synthetic 3D maps generated using canonical ASL kinetic forward model
Paper PLDs: 6 PLDs
Our PLDs: 4 PLDs (1.525, 2.025, 2.525, 3.025)
Paper loss: MAE (or equivalent map-wise loss)
Our loss: Masked Brain + Background MAE (equally weighted)
Paper training: Specific hyperparameter schedule for in-vivo data
Our training: 50 epochs, Adam optimizer, early stopping patience 15, learning rate 5e-4
Exact reproduction or adaptation: Adaptation
Reason for adaptation: The original network used 6 PLDs and high-resolution real data. We adapted the network width and spatial capacity (32x32x16) to fit CPU memory/compute constraints and match our synthetic 4-PLD research objective.
