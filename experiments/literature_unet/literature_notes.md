1. Reference: Luciw et al. 2022, DOI: 10.1002/mrm.29193
2. Original paper: 6 PLDs, 3D real ASL data, 3D U-Net
3. Our adaptation: 4 PLDs, synthetic simulated data, same U-Net concept
4. Key differences clearly stated
5. Normalization difference: paper used 95th percentile; we use channel-wise mean/std
6. Dataset difference: paper used real human brain scans; we use synthetic spatial maps
7. PLD difference: paper used 6 PLDs; we use 4
8. Statement: 'The implementation is inspired by the U-Net architecture described by Luciw et al. (2022), but adapted to four simulated PLDs and the present synthetic ASL forward model.'
