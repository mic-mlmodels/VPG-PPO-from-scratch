# %%
# 0
import torch
import torch.nn as nn
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode=None)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n


class PolicyNN:
    def __init__(self) -> None:
        self.layers = nn.Sequential([nn.Linear()])
