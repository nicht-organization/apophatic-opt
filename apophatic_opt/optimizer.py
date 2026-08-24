import numpy as np


class ApophaticOptimizer:
    """Apophatic Optimizer: Subtraktive Dämpfung (¬X) & Baseline-Relaxation (B_0)."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        sigma_threshold: float = 0.1,
        tolerance: float = None,
        gamma_relaxation: float = 0.001,
    ):
        self.lr = learning_rate
        self.sigma = tolerance if tolerance is not None else sigma_threshold
        self.gamma = gamma_relaxation

    def step(
        self, weights: np.ndarray, gradients: np.ndarray = None
    ) -> tuple[np.ndarray, dict]:
        if gradients is None:
            gradients = weights.copy()

        initial_PA = float(np.linalg.norm(gradients))

        # Subtraktiver Filter (Maskierung über Schwelle sigma)
        mask = np.abs(gradients) > self.sigma
        cleaned_gradients = np.where(mask, gradients, 0.0)

        # Update mit Dämpfung + Hard Threshold Collapse auf B_0
        relaxed_weights = (
            weights - self.lr * cleaned_gradients - self.gamma * weights
        )
        relaxed_weights *= mask  # In-Place Zeroing von Rauschen

        # Metriken
        final_PA = float(np.linalg.norm(cleaned_gradients))
        metrics = {
            "PA_reduction": initial_PA - final_PA,
            "friction_W": float(np.linalg.norm(relaxed_weights)),
            "sparsity": float(np.sum(~mask) / gradients.size),
        }

        return relaxed_weights, metrics