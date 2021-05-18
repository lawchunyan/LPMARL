import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch

from datetime import date
from env_wrapper.sc2_env_wrapper import StarCraft2Env
from src.agents.Qmixagent import QAgent

use_wandb = True
test = False
map_name='3m'
env = StarCraft2Env(map_name=map_name, window_size_x=400, window_size_y=300, enemy_obs=False)

state_dim = env.get_obs_size()
action_dim = env.n_actions - 1

# num_episodes = 200000  # goal: 2 million timesteps; 15000 episodes approx.
num_timestep = 2000000

agent_config = {'state_dim': state_dim,
                'action_dim': action_dim,
                'n_ag': env.n_agents,
                'memory_len': 500,
                'batch_size': 50,
                'train_start': 100,
                'epsilon_start': 1.0,
                'epsilon_decay': 1e-4,
                'gamma': 0.99,
                'hidden_dim': 32,
                'mixer': True,
                'loss_ftn': torch.nn.MSELoss(),
                'lr': 1e-4,
                'state_shape': env.get_state_size(),
                'memory_type': 'ep'
                }

if test:
    n_agents = 10
else:
    n_agents = env.n_agents
    if use_wandb:
        wandb.init(project='optmarl', name=date.today().strftime("%Y%m%d") + '-Qmix_'+map_name, config=agent_config)

agent = QAgent(**agent_config)

rewards = []

t_env = 0

for e in range(100000):
    if not test:
        env.reset()

    terminated = False
    prev_avail_action = None
    episode_reward = 0
    ep_len = 0

    while not terminated:
        t_env += 1
        ep_len += 1
        if test:
            state = np.random.random((n_agents, state_dim))
            avail_actions = np.random.randint(2, size=(n_agents, 14))
        else:
            global_state_prev = env.get_state()
            state = env.get_obs()
            avail_actions = env.get_avail_actions()

        env_action, action = agent.get_action(state, avail_actions)

        if test:
            reward, terminated = 1, 0
            next_state = np.random.random((n_agents, state_dim))
        else:
            reward, terminated, _ = env.step(env_action)
            next_state = env.get_obs()
            global_state_next = env.get_state()
            prev_avail_action = avail_actions

        agent.push(state, action, reward, next_state, terminated, avail_actions, global_state_prev, global_state_next)
        episode_reward += reward

    if agent.can_fit():
        agent.fit()

    if e % 200 == 0:
        agent.update_target()

    if e % 2000 == 0:
        agent.save(e)

    print("EP:{}, R:{}".format(e, episode_reward))
    rewards.append(episode_reward)
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'killed_enemy': env.death_tracker_enemy.sum(),
                   'EP': e,
                   'timestep': ep_len})

    if t_env >= num_timestep:
        break

env.close()
