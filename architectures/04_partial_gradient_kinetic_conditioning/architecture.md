# Architecture 04: Partial Gradient Kinetic Conditioning

## Architecture Diagram
`	ext
                    ┌───────────────┐
Input PLDs ────────►│ Shared Encoder│
                    └───────┬───────┘
                            │
                            h
                            │
                    ┌───────▼────────┐
                    │ Kinetic Encoder│
                    └───────┬────────┘
                            │
                          z_kin
                       ┌────┴────┐
                       │         │
                       ▼         ▼
                   ATT Head   Gradient-scaled
                              z_kin
                                  │
                                  ▼
                            CBF Encoder
                                  │
                                  ▼
                              CBF Head
`
**Note:** CBF gradient → kinetic encoder = 0.5. The forward values of z_kin are NOT scaled; only the backward gradients are scaled.
