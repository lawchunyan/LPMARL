import torch

from src.agents.LPagent import RLAgent
from env_wrapper.sc2_env_wrapper import StarCraft2Env

env = StarCraft2Env(map_name="8m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()
n_agents = env.n_agents
n_enemies = env.n_enemies

agent = RLAgent(state_dim, n_agents, n_enemies, batch_size=1, epsilon_start=0.0)
agent.load_state_dict(torch.load('happy.th'))


for e in range(10):
    env.reset()

    terminated = False
    episode_reward = 0

    while not terminated:
        state = env.get_obs()
        avail_actions = env.get_avail_actions()

        action, chosen_action_logit_h, high_action, low_action = agent.get_action(state, n_agents, n_enemies,
                                                                                  avail_actions)

        reward, terminated, _ = env.step(action)
        next_state = env.get_obs()

        episode_reward += reward

    print("EP:{}, R:{}".format(e, episode_reward))
