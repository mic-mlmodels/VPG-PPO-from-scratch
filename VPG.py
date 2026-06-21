# %%
# 0
import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import numpy as np

env = gym.make("CartPole-v1", render_mode=None)
OBS_DIM = env.observation_space.shape[0]
ACT_DIM = env.action_space.n
HIDDEN_DIM = 8
EPISODE_NUM = 1000
DISCOUNT = 0.99


class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=OBS_DIM, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=ACT_DIM),
        )

    def forward(self, x):
        return self.layers(x)


class ValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=OBS_DIM, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=1),
        )

    def forward(self, x):
        return self.layers(x)


policy_v0 = PolicyNN()
value_v0 = ValueNN()
policy_optimiser = torch.optim.AdamW(lr=3e-4, params=policy_v0.parameters())
value_optimiser = torch.optim.AdamW(lr=3e-4, params=value_v0.parameters())
episode_rewards = []
mean_rewards = []
for episode in range(EPISODE_NUM):
    policy_optimiser.zero_grad()
    value_optimiser.zero_grad()
    states_lst = []
    actions_lst = []
    rewards_lst = []
    log_probs_lst = []
    future_rewards_lst = []
    state_value_lst = []
    np_state, info = env.reset()
    while True:
        states_lst.append(np_state)
        torch_state = torch.as_tensor(np_state, dtype=torch.float32)
        state_value = value_v0(torch_state)
        state_value_lst.append(state_value)
        action_logits = policy_v0(torch_state)
        action_probs = F.softmax(action_logits, dim=-1)
        action_distribution = torch.distributions.Categorical(probs=action_probs)
        selected_action = action_distribution.sample()
        scalar_action = selected_action.item()
        log_probs = action_distribution.log_prob(selected_action)
        log_probs_lst.append(log_probs)
        actions_lst.append(selected_action)
        np_state, reward, terminated, truncated, info = env.step(scalar_action)
        rewards_lst.append(reward)
        if truncated or terminated:
            break
    for i in range(len(rewards_lst)):
        entry = 0
        for exp, reward in enumerate(rewards_lst[i:]):
            entry += 0.99**exp * reward
        future_rewards_lst.append(entry)

    advantage_lst = torch.as_tensor(
        future_rewards_lst, dtype=torch.float32
    ) - torch.stack(state_value_lst)
    loss = -torch.mean(advantage_lst * torch.stack(log_probs_lst))
    episode_rewards.append(torch.sum(torch.as_tensor(rewards_lst)))
    loss.backward()
    policy_optimiser.step()
    value_optimiser.step()
for i in range(EPISODE_NUM // 50):
    mean_rewards.append(np.mean(episode_rewards[i * 50 : (i + 1) * 50]))
print(mean_rewards)
