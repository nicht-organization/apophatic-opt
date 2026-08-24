# apophatic-opt

> **Apophatic Gradient Optimization Engine**  
> Subtractive Noise Filtering (¬X), Assertion Pressure Reduction (P_A), and Baseline Relaxation (B_0).

[![PyPI version](https://badge.fury.io/py/apophatic-opt.svg)](https://badge.fury.io/py/apophatic-opt)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://unlicense.org/)

`apophatic-opt` implements a novel optimization paradigm grounded in the **Nicht-Theorie** framework. Rather than adding arbitrary model capacity or hyper-parameters to enforce convergence, the apophatic optimizer operates by direct noise subtraction, eliminating non-essential gradient fluctuations and minimizing systemic friction (W -> 0).

## Installation

```bash
pip install apophatic-opt
```

## Quickstart

```python
import numpy as np
from apophatic_opt import ApophaticOptimizer

# Initialize optimizer with subtractive threshold sigma
opt = ApophaticOptimizer(
    learning_rate=0.01, sigma_threshold=0.1, gamma_relaxation=0.001
)

weights = np.random.normal(0, 1, size=1000)
gradients = weights.copy()

# Perform optimization step
relaxed_weights, metrics = opt.step(weights, gradients)

print(
    f"PA Reduction: {metrics['PA_reduction']:.4f} | Sparsity: {metrics['sparsity']:.2%}"
)
```

## Core Benchmarks

The suite validates four core theoretical properties:
1. **Jevons' Rebound Cancellation**: Prevents systemic inflation beyond saturation boundaries.
2. **Gossen Pruning**: Kills sub-marginal weight branches where friction exceeds utility.
3. **Fiat Noise Isolation**: Separates structural substrate signals from entropy noise.
4. **Kolmogorov Bounds**: Proves zero friction (W = 0) under non-distorted conditions.
