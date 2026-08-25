from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apophatic_opt.optimizer import ApophaticOptimizer

def test_heavytail_drift_isolation():
    np.random.seed(42)
    steps = 500
    
    # Base Signal (kontinuierliches, leichtes Signal) + Fat-Tail Spikes (Cauchy-Noise)
    base_signal = 0.005 * np.ones(100)
    
    # Parameter-Initialisierung
    weights_apo = np.zeros(100)
    weights_sgd = np.zeros(100)
    
    lr = 0.01
    sigma = 0.02  # Schwelle für Rausch-Auslöschung
    gamma = 0.001
    
    opt_apo = ApophaticOptimizer(learning_rate=lr, sigma_threshold=sigma, gamma_relaxation=gamma)
    
    apo_history = []
    sgd_history = []
    
    for t in range(steps):
        # 90% kleines Hintergrundrauschen, 10% extreme Spikes (Heavy Tail)
        is_spike = np.random.rand(100) > 0.90
        noise = np.where(is_spike, np.random.standard_cauchy(100) * 0.5, np.random.normal(0, 0.005, 100))
        
        # Effektiver Gradient = Signal + verrauschter Drift
        grad = base_signal + noise
        
        # 1. Apophatic Update
        weights_apo, metrics = opt_apo.step(weights_apo, grad)
        
        # 2. Klassisches Standard-SGD Update (zum Vergleich)
        weights_sgd = weights_sgd - lr * grad
        
        apo_history.append(np.linalg.norm(weights_apo))
        sgd_history.append(np.linalg.norm(weights_sgd))
        
    final_apo_norm = np.linalg.norm(weights_apo)
    final_sgd_norm = np.linalg.norm(weights_sgd)
    
    print(f"\n🚀 Heavy-Tail Drift Benchmark ({steps} Steps):")
    print(f"   - Standard SGD Final Weight Norm: {final_sgd_norm:.4f} (Explosion durch Spikes)")
    print(f"   - Apophatic-Opt Final Weight Norm: {final_apo_norm:.4f} (Gedämpft auf Basislinie)")
    print(f"   - Final Sparsity (Apophatic): {metrics['sparsity']*100:.2f}%")
    
    # Validierung: SGD muss durch die Spikes aufgebläht sein, Apophatic bleibt stabil
    assert final_apo_norm < final_sgd_norm, "Apophatic-Opt konnte den Drift nicht dämpfen!"
    assert metrics['sparsity'] > 0.5, "Sparsity unter Heavy-Tail Drift zu niedrig"

if __name__ == "__main__":
    test_heavytail_drift_isolation()
