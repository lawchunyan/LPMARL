import numpy as np
import matplotlib.pyplot as plt
import torch.nn

import wandb

from datetime import date
from env_wrapper.sc2_env_wrapper import StarCraft2Env
from src.agents.LPagent import RLAgent

TRAIN = True

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()

agent_config = {"state_dim": state_dim,
                "n_ag": env.n_agents,
                "n_en": env.n_enemies,
                "action_dim": 5,
                "memory_len": 500,
                "batch_size": 50,
                "train_start": 100,
                "epsilon_start": 1.0,
                "epsilon_decay": 5e-5,
                "gamma": 0.99,
                "hidden_dim": 32,
                "loss_ftn": torch.nn.MSELoss(),
                "lr": 5e-4,
                'memory_type': 'ep'
                }

n_agents = env.n_agents
n_enemies = env.n_enemies

agent = RLAgent(**agent_config)
agent.load_state_dict(torch.load('result/20210517_0.th'))

for e in range(10):
    env.reset()

    terminated = False
    episode_reward = 0
    t_env = 0

    while not terminated:
        state = env.get_obs()
        avail_actions = env.get_avail_actions()

        action, high_action, low_action = agent.get_action(state, avail_actions)

        reward, terminated, _ = env.step(action)
        next_state = env.get_obs()
        episode_reward += reward

env.close()
