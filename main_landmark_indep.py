import torch
import wandb
import os

from datetime import date
from src.agents.independent_ag import IndependentDDPGAgent
from src.utils.make_graph import make_graph
from envs.cooperative_navigation import make_env, get_landmark_state

TRAIN = True
use_wandb = True

n_ag = 5
num_episodes = 50000
coeff = 1.5
max_t = 50

agent_config = {
    "state_dim": 4 + 2 * (2 * n_ag - 1),
    "n_ag": n_ag,
    "n_en": n_ag,
    "action_dim": 5,
    "en_feat_dim": 2,
    'state_shape': (n_ag),
    "memory_len": 50000,
    "batch_size": 100,
    "train_start": 1000,
    "epsilon_start": 1.0,
    "epsilon_decay": 1e-6,
    "mixer": True,
    "gamma": 0.95,
    "hidden_dim": 32,
    "loss_ftn": torch.nn.MSELoss(),
    "lr": 0.01,
    'memory_type': 'sample',
    'target_tau': 0.05,
    'target_update_interval': 10,
    'coeff': coeff,
    "sc2": False
}

env = make_env(n_ag, n_ag)
agent = IndependentDDPGAgent(**agent_config)
agent_config['name'] = agent.name
print(agent.device)
agent.to(agent.device)

if TRAIN and use_wandb:
    exp_name = date.today().strftime("%Y%m%d") + "_navigation_" + agent_config['name'] + agent.device
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
    wandb.init(project='optmarl', name=exp_name, config=dict(agent_config, **exp_conf))
    wandb.watch(agent)

# agent.load_state_dict(torch.load('result/20210601_navigation_LP_2/4000.th'))

for e in range(num_episodes):
    episode_reward = 0
    state = env.reset()
    landmark_state = get_landmark_state(env)
    ep_len = 0
    std_action = 0

    while True:
        ep_len += 1
        if ep_len == 1:
            get_high_action = True
        else:
            get_high_action = False

        action, high_action, low_action = agent.get_action(state, landmark_state, explore=True,
                                                           get_high_action=get_high_action)
        # onehot_action = change_to_one_hot(action)

        std_action += action.std(axis=0)
        next_state, reward, terminated, _ = env.step(action)
        # next_state = make_graph(next_state, landmark_state)
        episode_reward += sum(reward)

        agent.push(state, landmark_state, action, reward, next_state, landmark_state)
        state = next_state

        if ep_len > max_t:
            break

    if agent.can_fit():
        ret_dict = agent.fit(e)
        wandb.log(ret_dict)

    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e,
                   'timestep': ep_len,
                   'num_hit': env.world.num_hit,
                   'std_action': std_action / ep_len})

    if e % 1000 == 0 and TRAIN:
        agent.save(curr_dir, e)

    for n in agent.noise:
        n.reset()

# if TRAIN:
#     agent.save(curr_dir, num_episodes)

env.close()
