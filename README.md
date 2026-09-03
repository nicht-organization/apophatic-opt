# apophatic-opt

> **Apophatic Gradient Optimization Engine**
> Subtractive Noise Filtering (¬X), Assertion Pressure Reduction (P_A), and Baseline Relaxation (B_0).

[![PyPI version](https://badge.fury.io/py/apophatic-opt.svg)](https://badge.fury.io/py/apophatic-opt)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://unlicense.org/)

**apophatic-opt** implements a novel optimization paradigm grounded in the **Nicht-Theorie** framework. Rather than adding arbitrary model capacity or hyper-parameters to enforce convergence, the apophatic optimizer operates by direct noise subtraction, eliminating non-essential gradient fluctuations and minimizing systemic friction (W -> 0).

## Installation

pip install apophatic-opt

## Quickstart

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

## Core Benchmarks

The suite validates four core theoretical properties:
1. **Jevons' Rebound Cancellation**: Prevents systemic inflation beyond saturation boundaries.
2. **Gossen Pruning**: Kills sub-marginal weight branches where friction exceeds utility.
3. **Fiat Noise Isolation**: Separates structural substrate signals from entropy noise.
4. **Kolmogorov Bounds**: Proves zero friction (W = 0) under non-distorted conditions.

## API Reference

### ApophaticOptimizer

class ApophaticOptimizer:
    def __init__(
        self,
        learning_rate: float = 0.01,
        sigma_threshold: float = 0.1,
        tolerance: float = None,
        gamma_relaxation: float = 0.001,
    ):

* **learning_rate** (eta): Step size scaling factor for surviving gradients.
* **sigma_threshold** (sigma): Subtractive gating threshold. Gradients with an absolute magnitude below this value are aggressively pruned to zero (B_0).
* **tolerance**: Backward-compatible alias for sigma_threshold.
* **gamma_relaxation** (gamma): Baseline relaxation coefficient governing continuous gravitational decay toward zero.

#### Methods
* **step(weights: np.ndarray, gradients: np.ndarray = None) -> tuple[np.ndarray, dict]**:
  Executes a single subtractive update cycle. If **gradients** is omitted, defaults to copying the current weight vector. Returns a tuple containing the updated **np.ndarray** weights and a telemetry dictionary:
  * **PA_reduction**: Magnitude reduction of the active gradient field.
  * **friction_W**: Residual systemic friction (W).
  * **sparsity**: Fraction of parameters successfully collapsed to baseline.

## Computational Complexity

* **Time Complexity**: O(N) per update step, where N is the total parameter dimension. Vectorized NumPy operations execute entirely in optimized C routines without iterative Python loops.
* **Space Complexity**: O(N) auxiliary memory allocation for transient Boolean masks and intermediate array transformations.

## Theoretical Details

* **The Apophatic Paradigm** (¬X): Borrowed from theological apophaticism—which defines absolute truth by negation—this optimizer models high-dimensional landscapes by systematically subtracting noise, speculative fiat fluctuations, and sub-marginal utility rather than adding parametric capacity.
* **Friction Suppression** (W -> 0): Bypasses traditional gradient explosion hazards and heavy-tail drift by enforcing a hard structural collapse to baseline (B_0) whenever local gradient energy drops beneath the empirical significance boundary sigma.
