import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from cliffeq.models.bottleneck_v2 import CliffordEPBottleneckV2
import os
import json
import time

# 1. HalfCheetah Z2 Mirror Symmetry
def get_mirror_mask(obs_dim):
    # Observation mirror mask for HalfCheetah-v4 (17D)
    # 0: rootz, 1: rootp (angle), 2-7: joint angles, 8: vx, 9: vz, 10: vp, 11-16: joint velocities
    # Mask negates angles and angular velocities.
    if obs_dim == 17:
        obs_mask = np.array([1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    else:
        obs_mask = np.ones(obs_dim)

    # Action mirror mask (6 actuators)
    act_mask = np.array([-1, -1, -1, -1, -1, -1])
    return obs_mask, act_mask

# 2. Custom Policy with Clifford Bottleneck
class CliffordPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        class CliffordExtractor(nn.Module):
            def __init__(self, feature_dim, last_layer_dim_pi, last_layer_dim_vf):
                super().__init__()
                self.latent_dim_pi = last_layer_dim_pi
                self.latent_dim_vf = last_layer_dim_vf

                # Shared features
                self.shared = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                )

                # Bottleneck for policy
                self.pi_bottleneck = CliffordEPBottleneckV2(
                    in_dim=128, out_dim=last_layer_dim_pi,
                    sig_g=torch.tensor([1.0, 1.0]), # Cl(2,0)
                    use_spectral_norm=True
                )

                # Bottleneck for value
                self.vf_bottleneck = CliffordEPBottleneckV2(
                    in_dim=128, out_dim=last_layer_dim_vf,
                    sig_g=torch.tensor([1.0, 1.0]),
                    use_spectral_norm=True
                )

            def forward(self, features):
                shared = self.shared(features)
                return self.pi_bottleneck(shared), self.vf_bottleneck(shared)

            # Stable-baselines3 calls these specifically
            def forward_actor(self, features):
                shared = self.shared(features)
                return self.pi_bottleneck(shared)

            def forward_critic(self, features):
                shared = self.shared(features)
                return self.vf_bottleneck(shared)

        self.mlp_extractor = CliffordExtractor(self.features_dim, 64, 64)

# 3. Experiment Runner
def run_pr1():
    print("PR1: Continuous Control with Geometric Policy on HalfCheetah-v4")
    env_id = "HalfCheetah-v4"

    # Check obs dim
    temp_env = gym.make(env_id)
    obs_dim = temp_env.observation_space.shape[0]
    print(f"Observation dimension: {obs_dim}")
    temp_env.close()

    # Training
    total_timesteps = 20000
    print(f"Training PPO with Clifford bottleneck for {total_timesteps} steps...")
    model = PPO(
        CliffordPPOPolicy,
        env_id,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        device="cpu" # Use CPU for stability in this environment
    )
    model.learn(total_timesteps=total_timesteps)

    # Evaluation
    print("\nEvaluating standard vs mirrored...")
    eval_env = gym.make(env_id)
    obs_mask, act_mask = get_mirror_mask(obs_dim)

    def evaluate(mirrored=False, n_episodes=5):
        total_rewards = []
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            truncated = False
            ep_reward = 0
            while not (done or truncated):
                if mirrored:
                    obs = obs * obs_mask

                action, _ = model.predict(obs, deterministic=True)

                if mirrored:
                    action = action * act_mask

                obs, reward, done, truncated, _ = eval_env.step(action)
                ep_reward += reward
            total_rewards.append(ep_reward)
        return np.mean(total_rewards), np.std(total_rewards)

    mean_std, std_std = evaluate(mirrored=False)
    mean_mir, std_mir = evaluate(mirrored=True)

    print(f"Standard Reward: {mean_std:.2f} +/- {std_std:.2f}")
    print(f"Mirrored Reward (Zero-Shot): {mean_mir:.2f} +/- {std_mir:.2f}")

    violation = abs(mean_std - mean_mir) / (max(abs(mean_std), abs(mean_mir)) + 1e-6)
    print(f"Mirror Equivariance Violation: {violation:.4f}")

    results = {
        "standard_reward": mean_std,
        "mirrored_reward": mean_mir,
        "violation": violation,
        "timesteps": total_timesteps
    }

    os.makedirs("results", exist_ok=True)
    with open("results/pr1_results.json", "w") as f:
        json.dump(results, f)
    print("PR1 Complete")

if __name__ == "__main__":
    run_pr1()
