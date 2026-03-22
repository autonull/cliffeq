import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import json
from cliffeq.models.hybrid import CliffordEPBottleneck, CliffordBPBottleneck
from cliffeq.energy.zoo import BilinearEnergy
from cliffeq.dynamics.rules import LinearDot

class Policy(nn.Module):
    def __init__(self, variant="baseline", hidden=64):
        super().__init__()
        self.variant = variant
        self.fc1 = nn.Linear(4, hidden)
        if variant == "clifford-ep":
            self.bottleneck = CliffordEPBottleneck(BilinearEnergy(16, 16, torch.tensor([1., 1.])), LinearDot(), comp=4)
            self.proj_back = nn.Linear(16, hidden)
        elif variant == "clifford-bp":
            self.bottleneck = CliffordBPBottleneck(hidden, 16, torch.tensor([1., 1.]))
        self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.variant == "clifford-ep":
            x = self.proj_back(self.bottleneck(x))
        elif self.variant == "clifford-bp":
            x = self.bottleneck(x)
        return F.softmax(self.fc2(x), dim=-1)

def run_ppo(variant):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    model = Policy(variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    for _ in range(15):
        s, _ = env.reset(); done = False; states, actions, rewards = [], [], []
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            probs = model(st); dist = torch.distributions.Categorical(probs)
            a = dist.sample(); states.append(st); actions.append(a); s, r, term, trunc, _ = env.step(a.item())
            rewards.append(r); done = term or trunc
        G = sum(rewards); loss = 0
        for st, a in zip(states, actions):
            loss -= torch.distributions.Categorical(model(st)).log_prob(a) * G
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Eval
    returns = []
    for _ in range(10):
        s, _ = env.reset(); done = False; ret = 0
        while not done:
            with torch.no_grad(): a = model(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item()
            s, r, term, trunc, _ = env.step(a); done = term or trunc; ret += r
        returns.append(ret)
    env.close()
    return np.mean(returns)

def main():
    print("RL Domain Ablation")
    results = {}
    for var in ["baseline", "clifford-ep", "clifford-bp"]:
        print(f"  Running {var}...")
        results[var] = run_ppo(var)
        print(f"    Return: {results[var]:.2f}")
    with open("results/p4_3_ppo_cartpole.json", "w") as f: json.dump(results, f)

if __name__ == "__main__": main()
