import torch.nn
import wandb
import os

from datetime import date
from envs.sc2_env_wrapper import StarCraft2Env
# from smac.env import StarCraft2Env
from src.agents.LPagent_hier_discrete import LPAgent

# from src.agents.Qmixagent import QmixAgent

TRAIN = True
use_wandb = True

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()
num_episodes = 2000000  # goal: 2 million timesteps; 15000 episodes approx.

agent_config = {"state_dim": state_dim,
                "n_ag": env.n_agents,
                "n_en": env.n_enemies,
                "action_dim": 5,
                "memory_len": 5000,
                "batch_size": 32,
                "train_start": 100,
                "epsilon_start": 1.0,
                "epsilon_decay": 1e-6,
                "gamma": 0.99,
                "hidden_dim": 32,
                "loss_ftn": torch.nn.MSELoss(),
                "lr": 5e-4,
                'memory_type': 'ep',
                'target_tau': 0.1,
                'target_update_interval': 200
                }

agent = LPAgent(**agent_config)
exp_name = date.today().strftime("%Y%m%d") + "_" + agent.name + '8m_group_v2'

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
    wandb.init(project='optmarl_sc2', name=exp_name, config=dict(agent_config, **exp_conf))

for e in range(num_episodes):
    env.reset()

    terminated = False
    episode_reward = 0
    ep_len = 0
    prev_killed_enemies = env.death_tracker_enemy.sum()
    high_action = None
    h_transition = True

    while not terminated:
        ep_len += 1
        state = env.get_obs()
        agent_obs = state[:env.n_agents]
        enemy_obs = state[env.n_agents:]
        avail_actions = env.get_avail_actions()

        if high_action is not None:
            h_transition = False

        action, high_action, low_action = agent.get_action(agent_obs, enemy_obs, avail_actions, high_action=high_action)

        reward, terminated, _ = env.step(action)

        next_killed_enemies = env.death_tracker_enemy.sum()
        next_state = env.get_obs()
        n_agent_obs = next_state[:env.n_agents]
        n_enemy_obs = next_state[env.n_agents:]

        high_r = 20 if env.death_tracker_enemy.sum() == env.n_enemies else 0
        reward = env.death_tracker_enemy[high_action.squeeze()] * 10 * (1 - env.death_tracker_ally)

        agent.push(agent_obs, enemy_obs, high_action, low_action, reward, n_agent_obs, n_enemy_obs, terminated,
                   avail_actions, high_r, h_transition)
        episode_reward += sum(reward)

        if prev_killed_enemies != next_killed_enemies:
            high_action = None
            h_transition = True

        prev_killed_enemies = next_killed_enemies

    if e % 500 == 0:
        agent.save(curr_dir, e)

    if agent.can_fit():
        agent.fit(e)

    print("EP:{}, R:{}".format(e, episode_reward))
    if use_wandb:
        wandb.log({'reward': high_r,
                   'reward_l': episode_reward,
                   'epsilon': agent.epsilon,
                   'killed_enemy': env.death_tracker_enemy.sum(),
                   'EP': e,
                   'timestep': ep_len})

env.close()
