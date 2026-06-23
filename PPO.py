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
EPISODE_NUM = 300
DISCOUNT = 0.99
EPSILON = 0.2
OLD_POLICY_LOOPS = 4
NEW_POLICY_LOOPS = 4


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
    if episode % 100 == 0:
        print(episode)
    old_policy_v0.load_state_dict(policy_v0.state_dict())
    for param in old_policy_v0.parameters():
        param.requires_grad = False
    with torch.no_grad():
        old_states_lst = []
        old_actions_lst = []
        old_log_probs_lst = []
        old_state_value_lst = []
        old_advantage_lst = []
        old_returns_lst = []
        old_temp_total_rewards = 0
        for i in range(OLD_POLICY_LOOPS):
            old_np_state, info = env.reset()
            old_rewards_lst = []
            while True:
                old_torch_state = torch.as_tensor(old_np_state, dtype=torch.float32)
                old_states_lst.append(old_torch_state)
                old_state_value = value_v0(old_torch_state.to(device))
                old_state_value_lst.append(old_state_value)
                old_action_logits = policy_v0(old_torch_state.to(device))
                old_action_probs = F.softmax(old_action_logits, dim=-1)
                old_action_distribution = torch.distributions.Categorical(
                    probs=old_action_probs
                )
                old_selected_action = old_action_distribution.sample()
                old_scalar_action = old_selected_action.item()
                old_log_probs = old_action_distribution.log_prob(old_selected_action)
                old_log_probs_lst.append(old_log_probs)
                old_actions_lst.append(old_selected_action)
                old_np_state, old_reward, old_terminated, old_truncated, old_info = (
                    env.step(old_scalar_action)
                )
                old_temp_total_rewards += 1
                old_rewards_lst.append(old_reward)
                if old_truncated or old_terminated:
                    break
            old_rewards_tensor = torch.tensor(
                old_rewards_lst, dtype=torch.float32, device=device
            )
            old_returns = torch.zeros_like(old_rewards_tensor, device=device)
            old_running_return = 0
            for t in reversed(range(len(old_rewards_lst))):
                old_running_return = old_rewards_lst[t] + DISCOUNT * old_running_return
                old_returns[t] = old_running_return
            old_returns_lst.append(old_returns)
    episode_rewards.append(old_temp_total_rewards / OLD_POLICY_LOOPS)
    old_returns_lst = torch.cat(old_returns_lst, dim=0)
    old_state_value_lst = torch.cat(old_state_value_lst, dim=-1)
    old_advantage_lst = old_returns_lst - old_state_value_lst
    for i in range(NEW_POLICY_LOOPS):
        policy_optimiser.zero_grad()
        value_optimiser.zero_grad()
        new_action_logits = policy_v0(torch.stack(old_states_lst).to(device))
        new_action_probs = F.softmax(new_action_logits, dim=-1)
        new_action_distributions = torch.distributions.Categorical(
            probs=new_action_probs
        )

        new_log_probs_lst = new_action_distributions.log_prob(
            torch.stack(old_actions_lst)
        )
        policy_loss = torch.mean(
            torch.minimum(
                torch.exp(new_log_probs_lst - torch.stack(old_log_probs_lst))
                * old_advantage_lst,
                torch.clip(
                    torch.exp(new_log_probs_lst - torch.stack(old_log_probs_lst)),
                    1 - EPSILON,
                    1 + EPSILON,
                )
                * old_advantage_lst,
            )
        )
        policy_loss.backward()
        new_state_value_lst = value_v0(torch.stack(old_states_lst).to(device))
        value_loss = torch.mean((new_state_value_lst - old_returns_lst) ** 2)
        value_loss.backward()
        policy_optimiser.step()
        value_optimiser.step()
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
