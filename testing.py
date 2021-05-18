import torch

from src.agents.LPagent import RLAgent
from env_wrapper.sc2_env_wrapper import StarCraft2Env

env = StarCraft2Env(map_name="3m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()
n_agents = env.n_agents
n_enemies = env.n_enemies
n_ep = 50

agent = RLAgent(state_dim, n_agents, n_enemies, batch_size=1, epsilon_start=0.0)
agent.load_state_dict(torch.load('result/20210518_19937.th'))
agent.epsilon_min = 0

win = 0
for e in range(n_ep):
    env.reset()

    terminated = False
    episode_reward = 0

    while not terminated:
        state = env.get_obs()
        avail_actions = env.get_avail_actions()

        action, high_action, low_action = agent.get_action(state, avail_actions)

        reward, terminated, _ = env.step(action)
        next_state = env.get_obs()

        episode_reward += reward

    print("EP:{:4}, R:{:>5}, Killed:{}".format(e, episode_reward.__round__(3), env.death_tracker_enemy.sum()))
    if env.death_tracker_enemy.sum() == env.n_enemies:
        win += 1

print('winnning ratio: {}'.format(win / n_ep))
env.close()
