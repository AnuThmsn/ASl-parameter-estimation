# Architecture 01: Shared Kinetic Representation

## 1. Research Hypothesis
Because CBF and ATT are both encoded in the same four-PLD ASL kinetic signal, learning a shared latent representation before parameter-specific prediction may improve CBF/ATT estimation compared with two completely independent networks.

## 2. Information-Analysis Motivation
The information-content validation demonstrated that CBF acts as a pure linear scalar, while ATT acts as a scale shift in early regimes and a shape modifier at PLD boundaries. A shared representation explicit models this dependency, forcing the network to understand the overall kinetic regime (ATT) to appropriately decode the scale (CBF), rather than independently trying to guess scale and shape.

## 3. Input
4 standardized ASL measurements (`[S1, S2, S3, S4]`)

## 4. Exact Layer-by-Layer Architecture
- **Shared Encoder:** 6 hidden layers (100 neurons each)
- **Shared Latent Representation:** 100-dimensional vector
- **ATT Head:** 3 hidden layers (100 neurons each) -> 1 output
- **CBF Head:** 3 hidden layers (50 neurons each) -> 1 output

## 5. Activation Functions
ELU (Exponential Linear Unit) for all hidden layers.

## 6. Shared Layers
The first 6 layers map the 4-dimensional normalized input into a 100-dimensional latent representation. This forces the model to extract common features required by both ATT and CBF.

## 7. CBF Head
Takes the 100-D shared latent vector and passes it through 3 layers of 50 neurons to predict standardized CBF.

## 8. ATT Head
Takes the 100-D shared latent vector and passes it through 3 layers of 100 neurons to predict standardized ATT.

## 9. Output Transformation
The network outputs standardized targets ($z_{CBF}, z_{ATT}$). These are un-standardized:
$CBF_{phys} = z_{CBF} \cdot \sigma_{CBF} + \mu_{CBF}$
and clamped to physical bounds: `CBF \in [0, 100]`, `ATT \in [0.5, 3.0]`.

## 10. Loss Function
$L = \text{MAE}(z_{CBF}^{pred}, z_{CBF}^{true}) + \text{MAE}(z_{ATT}^{pred}, z_{ATT}^{true})$

## 11. Number of Parameters
- **Shared Encoder:** ~51,000
- **CBF Head:** ~10,201
- **ATT Head:** ~30,401
- **Total:** ~91,602
*(Comparable to the independent baseline which has ~102,102 parameters).*

## 12. Training Protocol
- **Optimizer:** Adam (LR = 1e-3)
- **Batch Size:** 512
- **Early Stopping:** Patience of 10 epochs
- **Maximum Epochs:** 30 (Smoke test configuration)
- **Data:** Uniformly sampled CBF/ATT grid with dual-Rician noise.

## 13. Why Shared Representation is Being Tested
To isolate the effect of *weight sharing*. We want to know if forcing the network to learn a single unified kinetic curve representation improves its ability to disentangle scale from shape, particularly in the ill-conditioned long-ATT regimes.

## 14. What this architecture does NOT attempt to solve
It does not attempt to explicitly solve the fundamental ambiguity discovered in the information analysis via structural priors, nor does it explicitly condition CBF on ATT in a hierarchical chain. It merely tests if sharing early feature extraction helps.

## Diagram

```text
4 PLD ASL measurements
        |
        v
  Normalized input
        |
        v
 Shared FC encoder (6 layers x 100)
        |
        v
 Shared latent representation (100-D)
       / \
      /   \
     v     v
 ATT head CBF head
(3x100)   (3x50)
   |         |
   v         v
  ATT       CBF
```
