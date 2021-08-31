import matplotlib.pyplot as plt

import torch
import torch.nn

from envs.sc2_env_wrapper import StarCraft2Env
from src.agents.LPagent_Hier import LPAgent

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)
num_episodes = 1000

state_dim = env.get_obs_size()
agent_config = {"state_dim": state_dim,
                "n_ag": env.n_agents,
                "n_en": env.n_enemies,
                "action_dim": 5,
                "memory_len": 300,
                "batch_size": 50,
                "train_start": 100,
                "epsilon_start": 0,
                "epsilon_decay": 2e-5,
                "gamma": 0.99,
                "hidden_dim": 32,
                "loss_ftn": torch.nn.MSELoss(),
                "lr": 5e-4,
                'memory_type': 'ep',
                'target_tau': 0.5,
                'target_update_interval': 200
                }

agent = LPAgent(**agent_config)
agent.load_state_dict(torch.load("result/20210518_LPLeakyRELU/13995.th"))
agent.epsilon_min = 0.0
agent.epsilon = 0

n_win = 0
for e in range(num_episodes):
    env.reset()

    terminated = False
    episode_reward = 0
    ep_len = 0
    prev_killed_enemies = env.death_tracker_enemy.sum()

    while not terminated:
        ep_len += 1
        state = env.get_obs()
        agent_obs = state[:env.n_agents]
        enemy_obs = state[env.n_agents:]

        avail_actions = env.get_avail_actions()

        action, high_action, low_action = agent.get_action(agent_obs, enemy_obs, avail_actions, explore=False)

        reward, terminated, _ = env.step(action)
        episode_reward += reward

    if env.death_tracker_enemy.sum() > 2:
        n_win =+ 1

print(n_win/num_episodes)