import torch.nn
import wandb
import os

from datetime import date
from envs.combinatorial_env import ZeroShotGroupingEnv
from src.agents.LPagent_zeroshot import RLAgent
from src.agents.Qmixagent_zeroshot import QmixAgent
from src.agents.DQNAgent import DQNAgent

TRAIN = True
use_wandb = True

num_groups = 10
coeff = 4

env = ZeroShotGroupingEnv(n_ag=10, n_en=10, enemy_obs=True, global_reward=True, num_optimal_groups=num_groups)

state_dim = env.get_obs_size()
num_episodes = 20000  # goal: 2 million timesteps; 15000 episodes approx.

agent_config = {"state_dim": state_dim,
                "n_ag": env.n_agents,
                "n_en": env.n_enemies,
                'state_shape': (env.n_agents),
                "action_dim": env.n_enemies,
                "memory_len": 300,
                "batch_size": 50,
                "train_start": 100,
                "epsilon_start": 1.0,
                "epsilon_decay": 2e-4,
                "mixer": True,
                "gamma": 0.99,
                "hidden_dim": 32,
                "loss_ftn": torch.nn.MSELoss(),
                "lr": 5e-4,
                'memory_type': 'sample',
                'target_tau': 0.5,
                'target_update_interval': 200,
                'coeff': coeff
                }

agent = RLAgent(**agent_config)
agent_config['name'] = agent.name

if TRAIN:
    exp_name = date.today().strftime("%Y%m%d") + "_{}into{}_max{}_".format(env.n_agents, num_groups, coeff) + \
               agent_config['name']

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

    exp_conf = {'directory': curr_dir,
                'num_groups': num_groups}

    if use_wandb:
        wandb.init(project='optmarl', name=exp_name, config=dict(agent_config, **exp_conf))

else:
    use_wandb = False
    agent.load_state_dict(torch.load('result/20210524_10into10_max2_DQN/12000.th'))

for e in range(num_episodes):
    env.reset()

    terminated = False
    episode_reward = 0
    ep_len = 0

    while not terminated:
        ep_len += 1
        state = env.get_obs()

        action = agent.get_action(state)

        reward, terminated, _ = env.step(action)

        # next_state = env.get_obs()
        agent.push(state, action, reward)
        episode_reward += reward

    if e % 2000 == 0 and TRAIN:
        agent.save(curr_dir, e)

    if agent.can_fit():
        agent.fit(e)

    # print("EP:{}, R:{}".format(e, episode_reward))
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e,
                   'timestep': ep_len})

if TRAIN:
    agent.save(curr_dir, num_episodes)

env.close()
