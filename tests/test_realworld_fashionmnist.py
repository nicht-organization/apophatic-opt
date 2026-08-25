from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apophatic_opt.optimizer import ApophaticOptimizer

class SparseNet(nn.Module):
    def __init__(self):
        super(SparseNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    return (zero_params / total_params) * 100.0

def test_fashion_mnist_benchmark():
    device = torch.device("cpu")
    print(f"\n⚙️ Nutze Device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    model = SparseNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # KORRIGIERTE HYPERPARAMETER FÜR NEURONALE NETZE:
    # sigma_threshold stark gesenkt (1e-4), damit echte Gradienten durchgelassen werden
    optimizer = ApophaticOptimizer(
        learning_rate=0.05, 
        sigma_threshold=1e-4, 
        gamma_relaxation=0.0001
    )

    print("🚀 Starte Real-World Benchmark on Fashion-MNIST (2 Epochen)...")
    
    for epoch in range(1, 3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, target)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        w_np = param.data.numpy()
                        g_np = param.grad.data.numpy()
                        
                        new_w, metrics = optimizer.step(w_np, g_np)
                        
                        param.data.copy_(torch.from_numpy(new_w))

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        sparsity = calculate_sparsity(model)
        
        print(f"Epoch {epoch}: Test Accuracy = {accuracy:.2f}% | Sparsity = {sparsity:.2f}%")
        
    assert accuracy > 30.0, f"Accuracy Baseline unterschritten: {accuracy}%"

if __name__ == "__main__":
    test_fashion_mnist_benchmark()
