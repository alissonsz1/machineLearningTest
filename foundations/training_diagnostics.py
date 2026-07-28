import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        h = x
        with torch.no_grad():
            for layer in model.children():
                h = layer(h)
                if isinstance(layer, nn.Linear):
                    mean = h.mean().item()
                    std = h.std().item()
                    dead_fraction = (h <= 0).all(dim=0).float().mean().item()
                    stats.append({
                        "mean": round(mean,4),
                        "std": round(std,4),
                        "dead_fraction": round(dead_fraction,4)
                    })
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        stats = []

        model.zero_grad()
        pred = model(x)
        loss = nn.MSELoss()(pred,y)
        loss.backward()

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad.detach()
                mean = grad.mean().item()
                std = grad.std().item()
                norm = grad.norm().item()
                stats.append({
                    "mean": round(mean,4),
                    "std": round(std,4),
                    "norm": round(norm,4)
                })

        return stats
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"
            if stat["std"] < 0.1:
                return "vanishing_gradients"
            if stat["std"] > 10.0:
                return "exploding_gradients"
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

            
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        return "healthy"        


        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        pass
