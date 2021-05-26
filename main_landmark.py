from envs.cooperative_navigation import make_env

n_ag = 10

env_config = {'num_agents': n_ag,
              'num_enemies': n_ag}

exp_config = {'n_episodes': 100,
              }

agent_config = {"state_dim": 4 + 2 * (2 * n_ag - 1)}

env = make_env()
