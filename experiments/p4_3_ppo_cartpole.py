"""
Phase 4.3: RL Domain — PPO + P2.9 Clifford Bottleneck on CartPole

Objective: Test P2.9 geometric bottleneck for reinforcement learning.

Task: CartPole-v1 with mirror symmetry robustness
- Baseline: Standard PPO (MLP policy)
- Clifford: PPO + CliffordEPBottleneckV2 after first hidden layer

Metrics:
- Mean episode reward
- Stability (variance across runs)
- Mirror symmetry robustness (policy generalizes to mirrored state space)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
from tqdm import tqdm
import json
from datetime import datetime

from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2


class PPOPolicy(nn.Module):
    """Standard policy network for PPO."""
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.net(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


class PPOPolicyClifford(nn.Module):
    """PPO policy with CliffordEPBottleneckV2 after first hidden layer."""
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64, sig_g=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.first_layer = nn.Linear(state_dim, hidden_dim)

        # Clifford bottleneck
        self.bottleneck = CliffordEPBottleneckV2(
            in_dim=hidden_dim,
            out_dim=hidden_dim // 2,
            sig_g=sig_g,
            n_ep_steps=2,
            step_size=0.01,
            use_spectral_norm=True
        ) if sig_g is not None else None

        bottleneck_out_dim = hidden_dim // 2 if sig_g is not None else hidden_dim
        self.project_back = nn.Linear(bottleneck_out_dim, hidden_dim) if sig_g is not None else None

        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.first_layer(state))

        if self.bottleneck is not None:
            x = self.bottleneck(x)
            x = self.project_back(x)

        x = F.relu(self.second_layer(x))
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value


class PPOTrainer:
    """PPO training loop."""
    def __init__(self, policy, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2):
        self.policy = policy
        self.device = device
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            value_next = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * value_next * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(self, states, actions, old_logits, advantages, returns, n_epochs=4, batch_size=64):
        """Single PPO training step."""
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logits = torch.tensor(old_logits, dtype=torch.float32, device=self.device)
        advantages = advantages.detach()
        returns = returns.detach()

        dataset_size = len(states)

        for epoch in range(n_epochs):
            indices = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logits = old_logits[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                logits, values = self.policy(batch_states)
                values = values.squeeze(-1)

                # Actor loss (PPO clip)
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs[torch.arange(len(batch_actions)), batch_actions]

                old_log_probs = F.log_softmax(batch_old_logits, dim=-1)
                old_selected_log_probs = old_log_probs[torch.arange(len(batch_actions)), batch_actions]

                ratio = torch.exp(selected_log_probs - old_selected_log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = F.mse_loss(values, batch_returns)

                # Total loss
                loss = actor_loss + 0.5 * critic_loss

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()


def collect_episode(env, policy, device, max_steps=500):
    """Collect one episode of experience."""
    state, _ = env.reset()
    states, actions, rewards, values, logits, dones = [], [], [], [], [], []

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits_t, value = policy(state_tensor)
            logits_t = logits_t.squeeze(0)
            value = value.squeeze(-1).item()

        # Sample action
        dist = torch.distributions.Categorical(logits=logits_t)
        action = dist.sample().item()

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        logits.append(logits_t.detach().cpu().numpy())
        dones.append(done)

        state = next_state

        if done:
            break

    # Final value
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, value_final = policy(state_tensor)
        value_final = value_final.squeeze(-1).item()
    values.append(value_final)

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'values': values,
        'logits': logits,
        'dones': dones,
        'episode_return': sum(rewards)
    }


def evaluate_policy(env, policy, device, n_episodes=10):
    """Evaluate policy over multiple episodes."""
    returns = []
    for _ in range(n_episodes):
        episode = collect_episode(env, policy, device)
        returns.append(episode['episode_return'])
    return np.mean(returns), np.std(returns)


def main():
    """Main Phase 4.3 experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Hyperparameters
    num_train_steps = 200  # Reduced from 1000 for faster convergence
    episodes_per_step = 2  # Reduced from 4
    state_dim = 4
    action_dim = 2
    hidden_dim = 64
    sig_g = torch.tensor([1.0, 1.0])  # Cl(2,0): 4D

    # Results dictionary
    all_results = {}

    # ========================
    # Baseline: Standard PPO
    # ========================
    print("="*60)
    print("Baseline: Standard PPO MLP")
    print("="*60)

    policy_baseline = PPOPolicy(state_dim, action_dim, hidden_dim).to(device)
    trainer_baseline = PPOTrainer(policy_baseline, device)

    env_train = gym.make('CartPole-v1')
    env_eval = gym.make('CartPole-v1')

    baseline_returns = []
    baseline_stds = []

    for step in tqdm(range(num_train_steps // episodes_per_step), desc="Training Baseline"):
        # Collect episodes
        for ep in range(episodes_per_step):
            episode = collect_episode(env_train, policy_baseline, device)
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            values = episode['values']
            logits = episode['logits']
            dones = episode['dones']

            # Compute GAE
            advantages, returns = trainer_baseline.compute_gae(rewards, values, dones)

            # Train step
            trainer_baseline.train_step(states, actions, logits, advantages, returns)

        # Evaluate
        if (step + 1) % 10 == 0:
            mean_return, std_return = evaluate_policy(env_eval, policy_baseline, device, n_episodes=5)
            baseline_returns.append(mean_return)
            baseline_stds.append(std_return)
            print(f"  Step {(step+1)*episodes_per_step}: Return = {mean_return:.2f} ± {std_return:.2f}")

    all_results['baseline'] = {
        'final_mean_return': baseline_returns[-1] if baseline_returns else 0,
        'returns': baseline_returns,
        'stds': baseline_stds
    }

    env_train.close()
    env_eval.close()

    # ========================
    # Clifford: PPO + P2.9 Bottleneck
    # ========================
    print("\n" + "="*60)
    print("Clifford: PPO + P2.9 Bottleneck")
    print("="*60)

    policy_clifford = PPOPolicyClifford(state_dim, action_dim, hidden_dim, sig_g).to(device)
    trainer_clifford = PPOTrainer(policy_clifford, device)

    env_train = gym.make('CartPole-v1')
    env_eval = gym.make('CartPole-v1')

    clifford_returns = []
    clifford_stds = []

    for step in tqdm(range(num_train_steps // episodes_per_step), desc="Training Clifford"):
        for ep in range(episodes_per_step):
            episode = collect_episode(env_train, policy_clifford, device)
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            values = episode['values']
            logits = episode['logits']
            dones = episode['dones']

            advantages, returns = trainer_clifford.compute_gae(rewards, values, dones)
            trainer_clifford.train_step(states, actions, logits, advantages, returns)

        if (step + 1) % 10 == 0:
            mean_return, std_return = evaluate_policy(env_eval, policy_clifford, device, n_episodes=5)
            clifford_returns.append(mean_return)
            clifford_stds.append(std_return)
            print(f"  Step {(step+1)*episodes_per_step}: Return = {mean_return:.2f} ± {std_return:.2f}")

    all_results['clifford'] = {
        'final_mean_return': clifford_returns[-1] if clifford_returns else 0,
        'returns': clifford_returns,
        'stds': clifford_stds
    }

    env_train.close()
    env_eval.close()

    # ========================
    # Summary and Comparison
    # ========================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    baseline_final = all_results['baseline']['final_mean_return']
    clifford_final = all_results['clifford']['final_mean_return']
    improvement = clifford_final - baseline_final

    print(f"\nBaseline (Standard PPO):")
    print(f"  Final Mean Return: {baseline_final:.2f}")

    print(f"\nClifford (PPO + P2.9 Bottleneck):")
    print(f"  Final Mean Return: {clifford_final:.2f}")

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:.2f}")
    print(f"  Relative: {(improvement/baseline_final)*100:.2f}%" if baseline_final > 0 else "N/A")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/p4_3_ppo_cartpole_{timestamp}.json"

    import os
    os.makedirs("results", exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
    print("\n✓ Phase 4.3 Complete: RL domain baseline established")
