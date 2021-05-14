import numpy as np
import matplotlib.pyplot as plt
import torch.nn

import wandb

from datetime import date
from env_wrapper.sc2_env_wrapper import StarCraft2Env
from src.agents.LPagent import RLAgent

TRAIN = True
use_wandb = False

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()
num_episodes = 5000  # goal: 2 million timesteps; 15000 episodes approx.

agent_config = {"state_dim": state_dim,
                "n_ag": env.n_agents,
                "n_en": env.n_enemies,
                "action_dim": 5,
                "batch_size": 20,
                "memory_len": 5000,
                "epsilon_start": 1.0,
                "epsilon_decay": 2e-5,
                "train_start": 1000,
                "gamma": 0.99,
                "hidden_dim": 32,
                "loss_ftn": torch.nn.MSELoss(),
                "lr": 5e-4,
                'memory_type': 'ep'
                }

if not TRAIN:
    n_agents = 10
    n_enemies = 10
else:
    n_agents = env.n_agents
    n_enemies = env.n_enemies
    if use_wandb:
        wandb.init(project='optmarl', name=date.today().isoformat() + '-constraint4', config=agent_config)

agent = RLAgent(**agent_config)

rewards = []

for e in range(num_episodes):
    if TRAIN:
        env.reset()

    terminated = False
    episode_reward = 0

    while not terminated:
        if TRAIN:
            state = env.get_obs()
            avail_actions = env.get_avail_actions()
        else:
            state = np.random.random((n_agents + n_enemies, state_dim))
            avail_actions = np.random.randint(2, size=(n_agents, n_enemies + 5 + 2))

        action, high_action, low_action = agent.get_action(state, avail_actions)

        if TRAIN:
            reward, terminated, _ = env.step(action)
            next_state = env.get_obs()
        else:
            reward, terminated = 1, 0
            next_state = np.random.random((n_agents + n_enemies, state_dim))

        agent.push(state, high_action, low_action, reward, next_state, terminated, avail_actions)
        episode_reward += reward

    if agent.can_fit():
        agent.fit()

    print("EP:{}, R:{}".format(e, episode_reward))
    rewards.append(episode_reward)
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'killed_enemy': env.death_tracker_enemy.sum(),
                   'EP': e})

env.close()
agent.save()

plt.plot(rewards)
plt.savefig('rewards.png')
