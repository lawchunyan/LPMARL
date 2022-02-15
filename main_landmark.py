import torch
import wandb
import os
import numpy as np

from datetime import date
# from src.agents.LPagent_hier_maddpg import DDPGLPAgent
from src.agents.LPagent_hier_discrete import DDPGLPAgent
# from src.utils.make_graph import make_graph
from envs.cooperative_navigation import make_env, get_landmark_state, intrinsic_reward
from envs.normalize_rwd import reward_from_state

TRAIN = True
use_wandb = True

n_ag = 2
n_landmark = 2
num_episodes = 50000
coeff = 1.2
max_t = 50

agent_config = {
    "state_dim": 4 + 2 * (2 * n_ag - 1),
    "n_ag": n_ag,
    "n_en": n_ag,
    "action_dim": 5,
    "en_feat_dim": 2,
    'state_shape': (n_ag),
    "memory_len": 100000,
    "batch_size": 100,
    "train_start": 100,
    "epsilon_start": 1.0,
    "epsilon_decay": 1e-6,
    "mixer": True,
    "gamma": 0.95,
    "hidden_dim": 32,
    "loss_ftn": torch.nn.MSELoss(),
    "lr": 0.01,
    'memory_type': 'sample',
    'target_tau': 0.001,
    'target_update_interval': 50,
    'coeff': coeff,
    "sc2": False
}

env = make_env(n_ag, n_ag)
agent = DDPGLPAgent(**agent_config)
agent_config['name'] = agent.name
print(agent.device)
agent.to(agent.device)

n_fit = 0

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
    wandb.init(project='LPMARL_exp2', name=exp_name, config=dict(agent_config, **exp_conf), entity='curie')
    wandb.watch(agent)

# agent.load_state_dict(torch.load('result/20211110_navigation_ddpgcpu/7000.th'))

for e in range(num_episodes):
    episode_reward = 0
    episode_reward_l = 0

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

        # std_action += action.std(axis=0)
        next_state, _, terminated, _ = env.step(action)
        # next_state, reward, terminated, _ = env.step(action)
        rew_dense, n_occupied = reward_from_state(next_state)
        global_rwd = 10 if n_occupied == n_ag else 0

        # low_reward = [intrinsic_reward(env, i, a) for i, a in enumerate(high_action)]

        episode_reward_l += sum(rew_dense)
        episode_reward += global_rwd
        # reward = [sum(reward) / n_ag for r in reward]

        agent.push(state, landmark_state, high_action, low_action, rew_dense, next_state, landmark_state, terminated,
                   0,
                   global_rwd, get_high_action)
        state = next_state

        if all(terminated):
            break

        if agent.can_fit() and TRAIN:
            # for _ in range(4):
            n_fit += 1
            ret_dict = agent.fit(n_fit)
            wandb.log(ret_dict)
            # print(list(ret_dict.values()))

    if use_wandb:
        num_hit = 0
        for l in env.world.landmarks:
            landmark_pos = l.state.p_pos
            min_dist_to_l = 100
            for a in env.world.agents:
                ag_pos = a.state.p_pos
                min_dist_to_l = min(min_dist_to_l, np.sqrt(np.sum(np.square(landmark_pos - ag_pos))))

            if min_dist_to_l < 0.5:
                num_hit += 1
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e,
                   'num_hit': num_hit,
                   'std_action': std_action / ep_len,
                   'ep_len': ep_len,
                   'reward_l': episode_reward_l,
                   'coeff': coeff})

    if e % 1000 == 0 and e > 5000:
        agent.save(curr_dir, e)

    # for n in agent.noise:
    #     n.reset()

if TRAIN:
    agent.save(curr_dir, num_episodes)

env.close()
