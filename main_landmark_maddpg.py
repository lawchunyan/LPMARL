import os
import wandb

from src.agents.maddpg.maddpg import MADDPG
from envs.cooperative_navigation import make_env
from datetime import date

TRAIN = True
use_wandb = True

n_ag = 5
num_episodes = 50000
coeff = 1.5
max_t = 50

agent_config = {
    "state_dim": 4 + 2 * (2 * n_ag - 1),
    "n_ag": n_ag,
    # "n_en": n_ag,
    "action_dim": 5,
    # "en_feat_dim": 2,
    # 'state_shape': (n_ag),
    "memory_len": 50000,
    "batch_size": 50,
    "train_start": 1000,
    "epsilon_start": 1.0,
    "epsilon_decay": 1e-6,
    "gamma": 0.95,
    "hidden_dim": 32,
    "lr": 0.001,
    'target_tau': 0.005,
    # 'target_update_interval': 10,
}

env = make_env(n_ag, n_ag)
agent = MADDPG(**agent_config)
agent_config['name'] = agent.name
print(agent.device)
agent.to(agent.device)

if TRAIN and use_wandb:
    exp_name = date.today().strftime("%Y%m%d") + "_navigation_5ag_" + agent_config['name'] + agent.device
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

for e in range(num_episodes):
    episode_reward = 0
    state = env.reset()
    ep_len = 0

    while True:
        ep_len += 1

        action = agent.get_action(state, explore=True)
        next_state, reward, terminated, _ = env.step(action)

        episode_reward += sum(reward)

        agent.push(state, action, reward, next_state, terminated)
        state = next_state

        if ep_len > max_t:
            print("E:{}".format(e), "R:{:<5}".format(episode_reward))
            break

    agent.reset_noise()

    if len(agent.memory) > agent_config['train_start']:
        ret_dict = agent.fit(e)
        wandb.log(ret_dict)

    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'EP': e,
                   'num_hit': env.world.num_hit,
                   'epsilon': agent.agents[0].noise.epsilon})

    if e % 5000 == 0 and e > 1:
        agent.save(curr_dir, e)
