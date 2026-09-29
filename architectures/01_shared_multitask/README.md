# Proposed Next Architecture: Architecture 01 (Hierarchical Kinetic/Scale Network)

Based on the corrected information content analysis, we propose moving from independent CBF and ATT networks to a **Hierarchical Kinetic/Scale Architecture**.

## Scientific Justification
The analysis demonstrated that CBF and ATT are highly collinear at short ATTs ($ATT < PLD_1$). In this regime, changing CBF scales the whole four-vector, and changing ATT also acts primarily as a scale change. However, when ATT crosses the PLDs, shape information appears. 

Therefore, the measurements contain different kinds of information depending on the ATT regime. A generic "shared multitask network" is sub-optimal because the information structure is highly specific:
- CBF is a pure linear scale parameter.
- ATT dictates the kinetic/regime structure (the shape).

Our network should explicitly distinguish these two latent quantities, estimating the kinetic regime (ATT) first, and using that representation to untangle the amplitude (CBF).

## Proposed Design
```text
           [S1 S2 S3 S4]
                 │
        input normalization
                 │
           shared encoder
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
kinetic/regime       amplitude/scale 
representation       representation
      │                     │
      ▼                     │
     ATT                    │
      │                     │
      └─────────┬───────────┘
                ▼
          CBF estimation
```

## How to reproduce the information analysis:
From the repository root, run:
```bash
python analysis/01_information_content_4pld/validation_experiments.py
```
This will populate the `figures/` and `tables/` directories with the empirical data confirming this hypothesis.
