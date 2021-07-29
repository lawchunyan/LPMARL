import torch
import wandb
import os

from datetime import date
# from src.agents.LPagent_hier_maddpg import DDPGLPAgent
# from src.utils.make_graph import make_graph
from src.agents.maac.maac import AttentionSAC
from envs.cooperative_navigation import make_env, get_landmark_state, intrinsic_reward

TRAIN = True
use_wandb = True

n_ag = 3
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
    "memory_len": 50000,
    "batch_size": 100,
    "train_start": 100,
    "epsilon_start": 1.0,
    "epsilon_decay": 2e-6,
    "mixer": True,
    "gamma": 0.95,
    "hidden_dim": 32,
    "loss_ftn": torch.nn.MSELoss(),
    "lr": 0.01,
    'memory_type': 'sample',
    'target_tau': 0.005,
    'target_update_interval': 10,
    'coeff': coeff,
    "sc2": False
}

env = make_env(n_ag, n_ag)
agent = AttentionSAC.init_from_env(env, **agent_config)
agent_config['name'] = agent.name
print(agent.device)

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
    wandb.init(project='optmarl', name=exp_name, config=dict(agent_config, **exp_conf))

# agent.load_state_dict(torch.load('result/20210601_navigation_LP_2/4000.th'))

for e in range(num_episodes):
    episode_reward = 0
    episode_reward_l = 0

    state = env.reset()
    landmark_state = get_landmark_state(env)
    ep_len = 0
    std_action = 0

    while True:
        ep_len += 1

        torch_state = [torch.Tensor(state[i]).reshape(1, -1) for i in range(n_ag)]

        torch_action = agent.step(torch_state, explore=True)
        action = [a[0].data.numpy() for a in torch_action]

        next_state, reward, terminated, _ = env.step(action)

        episode_reward += sum(reward)

        agent.memory.push(state, action, reward, next_state, terminated)
        state = next_state

        if all(terminated):
            break

        if agent.can_fit() and TRAIN:
            n_fit += 1
            ret_dict = agent.fit(n_fit)
            wandb.log(ret_dict)

    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'EP': e,
                   'num_hit': env.world.num_hit,
                   'std_action': std_action / ep_len,
                   'reward_l': episode_reward_l})

    if e % 1000 == 0 and e > 5000:
        agent.save(curr_dir, e)

# if TRAIN:
#     agent.save(curr_dir, num_episodes)

env.close()
