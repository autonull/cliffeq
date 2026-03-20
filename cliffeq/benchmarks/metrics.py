import torch
import time
import json
import os
from typing import Callable, Any, Dict

def equivariance_violation(model: Callable, x: torch.Tensor, transform: Callable) -> float:
    with torch.no_grad():
        out_T = model(transform(x))
        T_out = model(x)
        diff = torch.norm(out_T - T_out) / (torch.norm(T_out) + 1e-8)
        return diff.item()

def convergence_curve(energy_fn: Callable, dynamics_rule: Any, x_init: torch.Tensor, n_steps: int, dt: float) -> list:
    energies = []
    x = x_init.detach().clone()
    for _ in range(n_steps):
        with torch.no_grad():
            energies.append(energy_fn(x).sum().item())
        x = dynamics_rule.step(x, energy_fn, dt)
    return energies

class MetricsLogger:
    def __init__(self, use_wandb: bool = False, json_path: str = "results/experiment.json"):
        self.use_wandb = use_wandb
        self.json_path = json_path
        self.results = []
        if use_wandb:
            import wandb
            self.wandb = wandb

    def log(self, metrics: Dict[str, Any]):
        self.results.append(metrics)
        if self.use_wandb:
            self.wandb.log(metrics)

    def save(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(self.results, f)

def run_experiment(config: Dict[str, Any], train_fn: Callable, eval_fn: Callable):
    logger = MetricsLogger(use_wandb=config.get("use_wandb", False), json_path=config.get("json_path", f"results/{config.get('name', 'exp')}.json"))
    start_time = time.time()
    train_results = train_fn(config)
    eval_results = eval_fn(config)
    end_time = time.time()
    summary = {
        "config": config,
        "train": train_results,
        "eval": eval_results,
        "total_time": end_time - start_time
    }
    logger.log(summary)
    logger.save()
    return summary
