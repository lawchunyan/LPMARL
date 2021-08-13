import wandb
import os

from datetime import date
from envs.cooperative_navigation import make_env
from src.agents.commnet.commnetagent import CommnetAgent

TRAIN = True
use_wandb = True

n_ag = 3
batch_size = 100
env = make_env(n_ag, n_ag)
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

agent = CommnetAgent(n_ag, state_dim, action_dim, batch_size)
agent.to(agent.device)

if TRAIN and use_wandb:
    exp_name = date.today().strftime("%Y%m%d") + "_navigation_" + 'commnet' + agent.device
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
    wandb.init(project='optmarl', name=exp_name)
    wandb.watch(agent)

for e in range(50000):
    ep_len = 0
    state = env.reset()
    episode_reward = 0
    while True:
        ep_len += 1
        action = agent.get_action(state, explore=True)
        ns, reward, terminated, _ = env.step(action)

        episode_reward += sum(reward)
        state = ns

        agent.push(state, action, reward, ns, terminated)

        if all(terminated):
            break

    if agent.can_fit():
        ret_dict = agent.fit()
        if use_wandb:
            wandb.log(ret_dict)

    if use_wandb:
        wandb.log({'reward': episode_reward,
                   'epsilon': agent.epsilon,
                   'EP': e,
                   'num_hit': env.world.num_hit,
                   'ep_len': ep_len
                   })
