# %%
# train
import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("CartPole-v1", render_mode=None)
OBS_DIM = env.observation_space.shape[0]  # type: ignore
ACT_DIM = env.action_space.n  # type: ignore
HIDDEN_DIM = 64
EPISODE_NUM = 5000
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


policy_v0 = PolicyNN().to(device)
old_policy_v0 = PolicyNN().to(device)
value_v0 = ValueNN().to(device)
policy_optimiser = torch.optim.AdamW(lr=3e-4, params=policy_v0.parameters())
value_optimiser = torch.optim.AdamW(lr=3e-4, params=value_v0.parameters())
episode_rewards = []
mean_rewards = []
for episode in range(EPISODE_NUM):
    old_policy_v0.load_state_dict(policy_v0.state_dict())
    for param in old_policy_v0.parameters():
        param.requires_grad = False
    policy_optimiser.zero_grad()
    value_optimiser.zero_grad()
    states_lst = []
    actions_lst = []
    rewards_lst = []
    log_probs_lst = []
    future_rewards_lst = []
    state_value_lst = []
    np_state, info = env.reset()
    if episode % 100 == 0:
        print(episode)
    while True:
        torch_state = torch.as_tensor(np_state, dtype=torch.float32)
        states_lst.append(torch_state)
        state_value = value_v0(torch_state.to(device))
        state_value_lst.append(state_value)
        action_logits = policy_v0(torch_state.to(device))
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
    old_action_logits = old_policy_v0(torch.stack(states_lst))
    old_action_probs = F.softmax(old_action_logits, dim=-1)
    old_action_distribution = torch.distributions.Categorical(probs=old_action_probs)
    old_log_probs_lst = old_action_distribution.log_prob(torch.stack(actions_lst))

    rewards_lst = torch.tensor(rewards_lst, dtype=torch.float32, device=device)
    returns = torch.zeros_like(rewards_lst, device=device)
    running_return = 0
    for t in reversed(range(len(rewards_lst))):
        running_return = rewards_lst[t] + DISCOUNT * running_return
        returns[t] = running_return
    advantage_lst = returns - torch.cat(state_value_lst, dim=-1)


for i in range(EPISODE_NUM // 50):
    mean_rewards.append(np.mean(episode_rewards[i * 50 : (i + 1) * 50]))
print(mean_rewards)

# %%
# display
import time

env = gym.make("CartPole-v1", render_mode="human")

state, info = env.reset()
done = False
truncated = False
total_reward = 0

while not (done or truncated):
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_logits = policy_v0(state_tensor)
        action = torch.argmax(action_logits).item()

    state, reward, done, truncated, info = env.step(action)
    total_reward += reward  # type: ignore
    time.sleep(0.05)

env.close()
