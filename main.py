import numpy as np
import matplotlib.pyplot as plt
import torch.nn

import wandb

from datetime import date
from env_wrapper.sc2_env_wrapper import StarCraft2Env
from src.agents.LPagent import RLAgent

TRAIN = True
use_wandb = True

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()
num_episodes = 20000  # goal: 2 million timesteps; 15000 episodes approx.

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
                'memory_type': 'ep',
                'target_tau':0.5
                }

if not TRAIN:
    n_agents = 10
    n_enemies = 10
else:
    n_agents = env.n_agents
    n_enemies = env.n_enemies
    if use_wandb:
        wandb.init(project='optmarl', name=date.today().strftime("%Y%m%d") + 'LPMARL', config=agent_config)

agent = RLAgent(**agent_config)

for e in range(num_episodes):
    if TRAIN:
        env.reset()

    terminated = False
    episode_reward = 0
    t_env = 0
    prev_killed_enemies = env.death_tracker_enemy.sum()

    while not terminated:
        t_env += 1
        state = env.get_obs()
        avail_actions = env.get_avail_actions()

        action, high_action, low_action = agent.get_action(state, avail_actions)

        reward, terminated, _ = env.step(action)

        next_killed_enemies = env.death_tracker_enemy.sum()
        next_state = env.get_obs()
        high_r = (next_killed_enemies - prev_killed_enemies) * 5

        agent.push(state, high_action, low_action, reward, next_state, terminated, avail_actions, high_r)
        episode_reward += reward
        prev_killed_enemies = next_killed_enemies

    if e % 2000 == 0 or episode_reward > 19.9:
        agent.save(e)

    if agent.can_fit():
        agent.fit()

    if e % 200 == 0:
        agent.update_target()



    print("EP:{}, R:{}".format(e, episode_reward))
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'killed_enemy': env.death_tracker_enemy.sum(),
                   'EP': e,
                   'timestep': t_env})

env.close()
