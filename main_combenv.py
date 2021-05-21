import torch.nn
import wandb
import os

from datetime import date
from envs.combinatorial_env import CombinatorialEnv
from src.agents.CombAgent import RLAgent

TRAIN = True
use_wandb = True

env = CombinatorialEnv(10, 10, enemy_obs=True, global_reward=True)

state_dim = env.get_obs_size()
num_episodes = 20000  # goal: 2 million timesteps; 15000 episodes approx.

agent_config = {"state_dim": state_dim,
                "n_ag": env.n_agents,
                "n_en": env.n_enemies,
                "action_dim": 5,
                "memory_len": 300,
                "batch_size": 50,
                "train_start": 100,
                "epsilon_start": 1.0,
                "epsilon_decay": 2e-5,
                "gamma": 0.99,
                "hidden_dim": 32,
                "loss_ftn": torch.nn.MSELoss(),
                "lr": 5e-4,
                'memory_type': 'sample',
                'target_tau': 0.5,
                'name': 'LP',
                'target_update_interval': 200
                }

agent = RLAgent(**agent_config)
if TRAIN:
    exp_name = date.today().strftime("%Y%m%d") + "_" + agent.name + "COenv"

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

else:
    use_wandb = False
    agent.load_state_dict(torch.load('result/20210521_LPCOenv_1/14000.th'))

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

    print("EP:{}, R:{}".format(e, episode_reward))
    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e,
                   'timestep': ep_len})

env.close()
