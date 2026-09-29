# Architecture 05: Adaptive Gradient Kinetic Conditioning

## Architecture Diagram
`	ext
                    ┌───────────────┐
Input PLDs ────────►│ Shared Encoder│
                    └───────┬───────┘
                            │
                            h
                            │
       ┌────────────────────┴────────────────────┐
       │                                         │
       ▼                                         ▼
┌───────────────┐                        ┌───────────────┐
│Kinetic Encoder│                        │ Gate Network  │
└───────┬───────┘                        └───────┬───────┘
        │                                        │
      z_kin                                   alpha(x)
        │                                        │
        ├──────────────────┐                     │
        │                  │                     │
        ▼                  ▼                     ▼
     ATT Head         Gradient-controlled CBF path
                      z_kin * alpha + z_kin.detach() * (1-alpha)
                           │
                           ▼
                      CBF Encoder
                           │
                           ▼
                       CBF Head
`
