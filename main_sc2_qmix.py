import wandb
import torch
import os

from datetime import date
from envs.sc2_env_wrapper import StarCraft2Env
from src.agents.Qmixagent import QmixAgent

use_wandb = True
test = False
map_name = '3m'
env = StarCraft2Env(map_name=map_name, window_size_x=400, window_size_y=300, enemy_obs=False)

state_dim = env.get_obs_size()
action_dim = env.n_actions - 1

# num_episodes = 200000  # goal: 2 million timesteps; 15000 episodes approx.
num_timestep = 5000000

agent_config = {'state_dim': state_dim,
                'action_dim': action_dim,
                'n_ag': env.n_agents,
                'memory_len': 500,
                'batch_size': 50,
                'train_start': 100,
                'epsilon_start': 1.0,
                'epsilon_decay': 5e-5,
                'gamma': 0.99,
                'hidden_dim': 32,
                'mixer': True,
                'loss_ftn': torch.nn.MSELoss(),
                'lr': 1e-4,
                'state_shape': env.get_state_size(),
                'memory_type': 'ep',
                'target_update_interval': 200,
                'target_tau': 0.5
                }

agent = QmixAgent(**agent_config)
exp_name = date.today().strftime("%Y%m%d") + "_" + agent.name

dirName = 'result/{}'.format(exp_name)
if os.path.exists(dirName):
    i = 0
    while True:
        i += 1
        curr_dir = dirName + "_{}".format(i)
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
            break

else:
    curr_dir = dirName
    os.makedirs(dirName)

exp_conf = {'directory': curr_dir}

if use_wandb:
    wandb.init(project='optmarl', name=exp_name, config=dict(agent_config, **exp_conf))

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

        global_state_prev = env.get_state()
        state = env.get_obs()
        avail_actions = env.get_avail_actions()

        env_action, action = agent.get_action(state, avail_actions)

        reward, terminated, _ = env.step(env_action)
        next_state = env.get_obs()
        global_state_next = env.get_state()
        prev_avail_action = avail_actions

        agent.push(state, action, reward, next_state, terminated, avail_actions, global_state_prev, global_state_next)
        episode_reward += reward

    if agent.can_fit():
        agent.fit(e)

    if e % 2000 == 0 or (episode_reward > 19.9 and e % 100 == 0):
        agent.save(curr_dir, e)

    print("EP:{}, R:{}".format(e, episode_reward))
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'killed_enemy': env.death_tracker_enemy.sum(),
                   'EP': e,
                   'timestep': ep_len})

    if t_env >= num_timestep:
        break

env.close()
