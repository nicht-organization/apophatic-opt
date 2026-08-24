from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apophatic_opt import ApophaticOptimizer


def test_jevons_paradox_rebound_cancellation():
    """Benchmark I: Jevons' Paradoxon & Rebound-Eliminierung."""
    opt = ApophaticOptimizer(
        learning_rate=0.01, sigma_threshold=0.05, gamma_relaxation=0.05
    )

    weights = np.ones(1000) * 0.5
    rebound_gradients = np.random.normal(0.03, 0.01, size=1000)

    relaxed_weights, metrics = opt.step(weights, rebound_gradients)

    assert (
        metrics["sparsity"] > 0.95
    ), "Jevons-Rebound wurde nicht abgefangen!"
    assert (
        metrics["friction_W"] < np.linalg.norm(weights)
    ), "Systemische Reibung hat durch Rebound zugenommen!"


def test_gossen_diminishing_returns_pruning():
    """Benchmark II: Grenznutzen-Kipppunkt (Gossensches Gesetz)."""
    opt = ApophaticOptimizer(
        learning_rate=0.1, sigma_threshold=0.1, gamma_relaxation=0.01
    )

    dim = 100
    diminishing_utility_grads = np.linspace(2.0, 0.01, dim)
    weights = np.ones(dim)

    relaxed_weights, metrics = opt.step(weights, diminishing_utility_grads)

    submarginal_indices = np.where(diminishing_utility_grads <= 0.1)[0]
    for idx in submarginal_indices:
        assert (
            abs(relaxed_weights[idx]) < weights[idx]
        ), f"Grenznutzen-Pruning fehlgeschlagen bei Index {idx}"


def test_fiat_noise_isolation():
    """Benchmark III: Spekulations-Rauschen Isolation."""
    dim = 10000
    np.random.seed(42)

    real_substrate = np.zeros(dim)
    real_indices = [10, 500, 2500, 8000]
    real_substrate[real_indices] = [15.0, 42.0, 8.5, 120.0]

    fiat_noise = np.random.uniform(-0.15, 0.15, size=dim)
    combined_system = real_substrate + fiat_noise

    # Realistische Lernrate 0.01 statt 1.0, damit Signale nicht in einem Schritt genullt werden
    opt = ApophaticOptimizer(
        learning_rate=0.01, sigma_threshold=0.2, gamma_relaxation=0.0
    )

    relaxed_system, metrics = opt.step(
        weights=combined_system, gradients=combined_system.copy()
    )

    # 1. Das Fiat-Rauschen unterhalb von sigma=0.2 kollabiert zu 100% auf B_0
    noise_only_indices = np.setdiff1d(np.arange(dim), real_indices)
    assert np.all(
        relaxed_system[noise_only_indices] == 0.0
    ), "Fiat-Rauschen wurde nicht vollständig isoliert!"

    # 2. Echte Substrat-Signale bleiben erhalten
    assert np.all(relaxed_system[real_indices] > 0.0)
    

def test_kolmogorov_friction_operator_bounds():
    """Benchmark IV: Formale Reibung W(P|I) & Kolmogorov-Grenzen."""
    opt = ApophaticOptimizer(
        learning_rate=0.01, sigma_threshold=0.01, gamma_relaxation=0.0
    )

    ideal_information = np.zeros(1000)
    _, metrics = opt.step(ideal_information, ideal_information)

    assert metrics["friction_W"] == 0.0
    assert metrics["PA_reduction"] == 0.0
    assert metrics["sparsity"] == 1.0