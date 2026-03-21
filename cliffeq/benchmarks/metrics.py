import torch
import time
import json
import os
from typing import Callable, Any, Dict, Optional

def equivariance_violation(model: Callable, x: torch.Tensor, transform_x: Callable, transform_y: Optional[Callable] = None) -> float:
    if transform_y is None:
        transform_y = transform_x
    with torch.no_grad():
        out_T = model(transform_x(x))
        T_out = transform_y(model(x))
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

def fixed_point_count(energy_fn: Callable, dynamics_rule: Any, shape: tuple, n_init: int = 200, dt: float = 0.1, n_steps: int = 100, eps: float = 1e-4) -> int:
    fixed_points = []
    for _ in range(n_init):
        x = torch.randn(shape) * 0.1
        for _ in range(n_steps):
            x = dynamics_rule.step(x, energy_fn, dt)

        is_new = True
        for fp in fixed_points:
            if torch.norm(x - fp) < eps:
                is_new = False
                break
        if is_new:
            fixed_points.append(x.detach().clone())
    return len(fixed_points)

def fixed_point_analysis(energy_fn: Callable, dynamics_rule: Any, shape: tuple, n_init: int = 100, dt: float = 0.1, n_steps: int = 100) -> Dict[str, Any]:
    final_energies = []
    converged_states = []
    for _ in range(n_init):
        x = torch.randn(shape) * 0.1
        for _ in range(n_steps):
            x = dynamics_rule.step(x, energy_fn, dt)
        with torch.no_grad():
            final_energies.append(energy_fn(x).sum().item())
        converged_states.append(x.detach())

    final_energies = torch.tensor(final_energies)
    return {
        "energy_mean": final_energies.mean().item(),
        "energy_std": final_energies.std().item(),
        "energy_min": final_energies.min().item(),
        "energy_max": final_energies.max().item(),
        "distinct_attractors": fixed_point_count(energy_fn, dynamics_rule, shape, n_init=n_init, dt=dt, n_steps=n_steps)
    }

class MetricsLogger:
    def __init__(self, use_wandb: bool = False, json_path: str = "results/experiment.json"):
        self.use_wandb = use_wandb
        self.json_path = json_path
        self.results = []
        if use_wandb:
            import wandb
            self.wandb = wandb

    def log(self, metrics: Dict[str, Any], energy_fn: Any = None):
        if energy_fn is not None and hasattr(energy_fn, "get_max_singular_value"):
            sn_val = energy_fn.get_max_singular_value()
            if sn_val is not None:
                metrics["max_singular_value"] = sn_val

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
