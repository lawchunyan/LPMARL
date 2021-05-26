import argparse
from collections import namedtuple
from envs.cooperative_navigation import make_env, get_landmark_state

n_ag = 10
n_ep = 100


env_config = {'num_agents': n_ag,
              'num_enemies': n_ag,
              }

# exp_config = {'n_episodes': n_ep,
#               }

agent_config = {"n_ag": n_ag,
                "n_en": n_ag,
                "action_dim": 5,
                "en_feat_dim": 2,
                "state_dim": 4 + 2 * (2 * n_ag - 1),
                "sc2": False,
            }

env = make_env()

for e in range(n_ep):
    t = 0
    state = env.reset()
    landmark_state = get_landmark_state(env)

    while True:
        t += 1
        action, high_action, low_action = agent.get_action(state, landmark_state)
        ns, reward, terminated, _ = env.step(action)

        agent.push(state, landmark_state, high_action, reward, next_state, terminated, 0, reward)