from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apophatic_opt.optimizer import ApophaticOptimizer

def test_kolmogorov_friction_limits():
    np.random.seed(42)
    steps = 300
    
    # Exaktes, rauschfreies Invarianten-Signal
    target_signal = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    weights_apo = np.zeros_like(target_signal)
    
    lr = 0.1
    sigma = 0.0     # Keine Auslöschung von Echtsignalen
    gamma = 0.001   # Minimale Kontinuierliche Baseline-Relaxation
    
    opt = ApophaticOptimizer(learning_rate=lr, sigma_threshold=sigma, gamma_relaxation=gamma)
    
    friction_history = []
    
    for t in range(steps):
        # Gradient ist die reine Abweichung vom Ziel-Signal (kein Rauschen!)
        grad = weights_apo - target_signal
        
        weights_apo, metrics = opt.step(weights_apo, grad)
        friction_history.append(metrics["friction_W"])
        
    # Rechnerische Gleichgewichtslage: w* = target_signal * (lr / (lr + gamma))
    expected_equilibrium = target_signal * (lr / (lr + gamma))
    reconstruction_error = np.linalg.norm(weights_apo - expected_equilibrium)
    
    print(f"\n🚀 Kolmogorov Friction Limit Benchmark ({steps} Steps):")
    print(f"   - Target Signal Norm: {np.linalg.norm(target_signal):.4f}")
    print(f"   - Reconstructed State Norm: {np.linalg.norm(weights_apo):.4f}")
    print(f"   - Theoretical Equilibrium Deviation: {reconstruction_error:.6e}")
    print(f"   - Final System Friction (W): {metrics['friction_W']:.4f}")
    
    # Validierung: Exakte Konvergenz gegen das theoretische Minimum ohne Oszillation
    assert reconstruction_error < 1e-4, "Reibungsfreie Rekonstruktion fehlgeschlagen!"

if __name__ == "__main__":
    test_kolmogorov_friction_limits()
