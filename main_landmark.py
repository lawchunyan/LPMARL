import torch
import wandb
import os
from datetime import date
from src.agents.LPagent_hier_maddpg import DDPGLPAgent
from envs.cooperative_navigation import make_env, get_landmark_state

TRAIN = True
use_wandb = True

n_ag = 5
num_episodes = 10000
coeff = 1.5
max_t = 50

agent_config = {
    "state_dim": 4 + 2 * (2 * n_ag - 1),
    "n_ag": n_ag,
    "n_en": n_ag,
    "action_dim": 5,
    "en_feat_dim": 2,
    'state_shape': (n_ag),
    "memory_len": 5000,
    "batch_size": 20,
    "train_start": 50,
    "epsilon_start": 1.0,
    "epsilon_decay": 2e-5,
    "mixer": True,
    "gamma": 0.95,
    "hidden_dim": 32,
    "loss_ftn": torch.nn.MSELoss(),
    "lr": 5e-4,
    'memory_type': 'ep',
    'target_tau': 0.005,
    'target_update_interval': 10,
    'coeff': coeff,
    "sc2": False
}

env = make_env(n_ag, n_ag)
agent = DDPGLPAgent(**agent_config)
agent_config['name'] = agent.name
print(agent.device)
agent.to(agent.device)

if TRAIN and use_wandb:
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
        action, high_action, low_action = agent.get_action(state, landmark_state, explore=True)
        # onehot_action = change_to_one_hot(action)

        std_action += action.std(axis=0)
        next_state, reward, terminated, _ = env.step(action)
        episode_reward += sum(reward)

        agent.push(state, landmark_state, high_action, low_action, reward, next_state, landmark_state, terminated, 0,
                   reward)
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

    if e % 200 == 0 or env.world.num_hit > 30:
        agent.save(curr_dir, e)

    for n in agent.noise:
        n.t = 0

if TRAIN:
    agent.save(curr_dir, num_episodes)

env.close()
