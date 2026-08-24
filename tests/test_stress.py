from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apophatic_opt import ApophaticOptimizer


def generate_noisy_logits(dim=50000, noise_level=0.05, signal_ratio=0.01):
    """Generiert hoch-dimensionale Logit-Landschaften für Colab-Free-Tier Bounds."""
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, size=dim)

    signal_indices = np.random.choice(
        dim, size=int(dim * signal_ratio), replace=False
    )
    noise[signal_indices] += np.random.uniform(
        1.0, 5.0, size=len(signal_indices)
    )
    return noise, signal_indices


def test_benchmark_noise_collapse_ratio():
    """Prüft, ob >90% des rein hoch-entropischen Rauschens auf B_0 kollabieren."""
    dim = 50000
    data, _ = generate_noisy_logits(
        dim=dim, noise_level=0.05, signal_ratio=0.01
    )

    # In tests/test_stress.py bei test_benchmark_noise_collapse_ratio:
    opt = ApophaticOptimizer(
        learning_rate=0.01, sigma_threshold=0.10, gamma_relaxation=0.001
    )

    relaxed_weights, metrics = opt.step(weights=data, gradients=data.copy())

    assert metrics["PA_reduction"] > 0.0, "Keine Druckreduktion gemessen"
    assert (
        metrics["sparsity"] > 0.90
    ), f"Sparsity zu gering: {metrics['sparsity']}"


def test_benchmark_signal_preservation():
    """Stellt sicher, dass echtes Signal trotz aggressiver Dämpfung intakt bleibt."""
    dim = 20000
    data, signal_idx = generate_noisy_logits(
        dim=dim, noise_level=0.03, signal_ratio=0.05
    )

    opt = ApophaticOptimizer(
        learning_rate=0.01, sigma_threshold=0.05, gamma_relaxation=0.0
    )
    relaxed_weights, _ = opt.step(weights=data, gradients=data.copy())

    signal_preserved = np.all(np.abs(relaxed_weights[signal_idx]) > 0.0)
    assert (
        signal_preserved
    ), "Signifikante Signale wurden versehentlich gedämpft!"