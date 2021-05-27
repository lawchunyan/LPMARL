import torch
import wandb
import os
from datetime import date
from utils.action_utils import change_to_one_hot
from src.agents.LPagent_Hier import LPAgent
from envs.cooperative_navigation import make_env, get_landmark_state

TRAIN = True
use_wandb = True

n_ag = 10
ep_len = 1000
coeff = 3
max_t = 30

agent_config = {
    "state_dim": 4 + 2 * (2 * n_ag - 1),
    "n_ag": n_ag,
    "n_en": n_ag,
    "action_dim": 5,
    "en_feat_dim": 2,
    'state_shape': (n_ag),
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
    'coeff': coeff,
    "sc2": False
}

env = make_env(n_ag, n_ag)
agent = LPAgent(**agent_config)
agent_config['name'] = agent.name

if TRAIN:
    exp_name = date.today().strftime("%Y%m%d") + "_navigation_" + agent_config['name']
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

for e in range(ep_len):
    t = 0
    episode_reward = 0
    state = env.reset()
    landmark_state = get_landmark_state(env)

    while True:
        t += 1
        action, high_action, low_action = agent.get_action(state, landmark_state)
        onehot_action = change_to_one_hot(action)
        ns, reward, terminated, _ = env.step(onehot_action)
        episode_reward += sum(reward)

        agent.push(state, landmark_state, high_action, low_action, reward, ns, landmark_state, terminated, 0, reward)

        if t > max_t:
            break

    if agent.can_fit():
        agent.fit(e)

    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e,
                   'timestep': ep_len})

if TRAIN:
    agent.save(curr_dir, ep_len)

env.close()
