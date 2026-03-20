import torch
from torch import nn
from typing import Callable, Optional

class FFEngine:
    def __init__(self, goodness_fn: Callable, threshold_theta: float):
        self.goodness_fn = goodness_fn
        self.threshold_theta = threshold_theta

    def train_layer(self, layer: nn.Module, positive_data: torch.Tensor, negative_data: torch.Tensor, optimizer: torch.optim.Optimizer, n_epochs: int = 1):
        for _ in range(n_epochs):
            optimizer.zero_grad()
            h_pos = layer(positive_data)
            h_neg = layer(negative_data)
            g_pos = self.goodness_fn(h_pos)
            g_neg = self.goodness_fn(h_neg)
            loss = torch.log(1 + torch.exp(-(g_pos - self.threshold_theta))).mean() + \
                   torch.log(1 + torch.exp(-(self.threshold_theta - g_neg))).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return layer(positive_data).detach(), layer(negative_data).detach()
