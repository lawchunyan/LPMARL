import numpy as np
import matplotlib.pyplot as plt
import wandb

from env_wrapper.sc2_env_wrapper import StarCraft2Env
from src.agents.Qmixagent import QAgent

use_wandb = False
test = True
env = StarCraft2Env(map_name="8m", window_size_x=400, window_size_y=300, enemy_obs=False)

state_dim = env.get_obs_size()
action_dim = env.n_actions - 1

num_episodes = 5000  # goal: 2 million timesteps; 15000 episodes approx.
if test:
    n_agents = 10
else:
    n_agents = env.n_agents
    if use_wandb:
        wandb.init(project='optmarl', name='constraint6')
batch_size = 20

agent = QAgent(state_dim, action_dim, n_agents, batch_size=batch_size)

rewards = []

for e in range(num_episodes):
    if not test:
        env.reset()

    terminated = False
    episode_reward = 0

    while not terminated:
        if test:
            state = np.random.random((n_agents, state_dim))
            avail_actions = np.random.randint(2, size=(n_agents, 14))
        else:
            state = env.get_obs()
            avail_actions = env.get_avail_actions()

        action = agent.get_action(state, avail_actions)

        if test:
            reward, terminated = 1, 0
            next_state = np.random.random((n_agents, state_dim))
        else:
            reward, terminated, _ = env.step(action)
            next_state = env.get_obs()

        agent.push(state, action, reward, next_state, terminated, avail_actions)
        episode_reward += reward

    if agent.can_fit():
        agent.fit()

    print("EP:{}, R:{}".format(e, episode_reward))
    rewards.append(episode_reward)
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e})

env.close()
agent.save()

plt.plot(rewards)
plt.savefig('rewards.png')
