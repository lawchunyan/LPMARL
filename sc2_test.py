import matplotlib.pyplot as plt

import torch
import torch.nn

from envs.sc2_env_wrapper import StarCraft2Env
from src.agents.LPagent_Hier import LPAgent

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)
num_episodes = 100

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
# agent.load_state_dict(torch.load("result/20210901_LP8m_group_v2_13/{}.th"))
agent.epsilon_min = 0.0
agent.epsilon = 0

# sd = range(54000, 55500, 500)
sd = [30000]

n_win = 0

for n in sd:
    agent.load_state_dict(torch.load("result/20210901_LP8m_group_v2_13/{}.th".format(n)))
    n_win = 0

    for e in range(num_episodes):
        env.reset()

        terminated = False
        episode_reward = 0
        ep_len = 0
        prev_killed_enemies = 0
        high_action = None

        while not terminated:
            state = env.get_obs()
            agent_obs = state[:env.n_agents]
            enemy_obs = state[env.n_agents:]
            avail_actions = env.get_avail_actions()

            action, high_action, low_action = agent.get_action(agent_obs, enemy_obs, avail_actions,
                                                               high_action=high_action, explore=False)
            reward, terminated, _ = env.step(action)

            next_killed_enemies = env.death_tracker_enemy.sum()



            if prev_killed_enemies != next_killed_enemies:
                print(len(set(high_action.tolist())))
                high_action = None

            prev_killed_enemies = next_killed_enemies

        if env.death_tracker_enemy.sum() >= 3:
            n_win += 1
        # n_win += env.death_tracker_enemy.sum()

    print(n, ":", n_win / num_episodes)
