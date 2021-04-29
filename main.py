import numpy as np
import matplotlib.pyplot as plt
import wandb

from env_wrapper.sc2_env_wrapper import StarCraft2Env
from RL.agent import RLAgent

test = False
env = StarCraft2Env(map_name="8m", window_size_x=400, window_size_y=300, enemy_obs=True)

state_dim = env.get_obs_size()
num_episodes = 5000  # goal: 2 million timesteps; 15000 episodes approx.
if test:
    n_agents = 10
    n_enemies = 10
else:
    n_agents = env.n_agents
    n_enemies = env.n_enemies
    wandb.init(project='optmarl', name='constraint6')
batch_size = 20

agent = RLAgent(state_dim, n_agents, n_enemies, batch_size=batch_size)

rewards = []

for e in range(num_episodes):
    if not test:
        env.reset()

    terminated = False
    episode_reward = 0

    while not terminated:
        if test:
            state = np.random.random((n_agents + n_enemies, state_dim))
            avail_actions = np.random.randint(2, size=(n_agents, n_enemies + 5 + 2))
        else:
            state = env.get_obs()
            avail_actions = env.get_avail_actions()

        action, chosen_action_logit_h, high_action, low_action = agent.get_action(state, n_agents, n_enemies,
                                                                                  avail_actions)

        if test:
            reward, terminated = 1, 0
            next_state = np.random.random((n_agents + n_enemies, state_dim))
        else:
            reward, terminated, _ = env.step(action)
            next_state = env.get_obs()

        agent.push(state, high_action, low_action, reward, next_state, terminated, avail_actions, chosen_action_logit_h)
        episode_reward += reward

    if agent.can_fit():
        agent.fit()

    print("EP:{}, R:{}".format(e, episode_reward))
    rewards.append(episode_reward)
    wandb.log({'reward': episode_reward,
               'epsilon': agent.epsilon,
               'EP': e})

env.close()
agent.save()

plt.plot(rewards)
plt.savefig('rewards.png')
