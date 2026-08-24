import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apophatic_opt import ApophaticOptimizer


def test_zero_vector_edge_case():
    """Null-Vektor darf keine NaNs oder Invarianz-Brüche erzeugen."""
    opt = ApophaticOptimizer(learning_rate=0.01, sigma_threshold=0.1)
    data = np.zeros(100)

    relaxed, metrics = opt.step(data, data)
    assert np.all(relaxed == 0.0)
    assert metrics["sparsity"] == 1.0


def test_apophatic_pruning_edge_case():
    """Prüft die exakte Grenzwert-Subtraktion an der Schwelle."""
    opt = ApophaticOptimizer(
        learning_rate=1.0, sigma_threshold=1e-3, gamma_relaxation=0.0
    )
    data = np.array([5.0, 0.0001, -0.0005, 1.2])

    relaxed, metrics = opt.step(weights=data, gradients=data.copy())

    # Werte unterhalb der Schwelle (0.0001 & -0.0005) kollabieren auf B_0 (0.0)
    assert math.isclose(relaxed[1], 0.0, abs_tol=1e-9)
    assert math.isclose(relaxed[2], 0.0, abs_tol=1e-9)