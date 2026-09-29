# Architecture 05: Adaptive Gradient Kinetic Conditioning

## Motivation
Fixed gradient coupling (0%, 50%, 100%) acts as a global optimization parameter, but physical identifiability varies wildly per sample. Architecture 05 tests if the network can learn a regime-dependent gradient routing policy.

## Mechanism
A small gate network predicts lpha(x) per-sample, which dynamically controls the magnitude of the CBF loss backpropagating into the kinetic encoder without altering the forward-pass representation.

## Check 
esults.md for full evaluation.
