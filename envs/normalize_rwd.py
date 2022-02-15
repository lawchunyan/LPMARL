import numpy as np


def reward_from_state_capacity(n_state, capacity):
    rew = []
    n_landmark = len(n_state)
    n_occupied = [0 for _ in range(n_landmark)]

    for state in n_state: # iterate on agent

        obs_landmark = np.array(state[4:4 + 2 * n_landmark])
        agent_reward = 0
        for i in range(n_landmark): # iterate on landmark

            sub_obs = obs_landmark[i * 2: i * 2 + 2]
            dist = np.sqrt(sub_obs[0] ** 2 + sub_obs[1] ** 2)

            if dist < 0.4: agent_reward += 0.3
            if dist < 0.2: agent_reward += 0.5
            if dist < 0.1:
                agent_reward += 10
                n_occupied[i] += 1
                break

        rew.append(agent_reward)

    return np.array(rew), sum(n_occupied)


def reward_from_state(n_state):
    rew = []
    n_ag = len(n_state)
    n_occupied = [False for _ in range(n_ag)]

    for state in n_state:

        obs_landmark = np.array(state[4:4 + 2 * n_ag])
        agent_reward = 0
        for i in range(n_ag):

            sub_obs = obs_landmark[i * 2: i * 2 + 2]
            dist = np.sqrt(sub_obs[0] ** 2 + sub_obs[1] ** 2)

            if dist < 0.4: agent_reward += 0.3
            if dist < 0.2:
                agent_reward += 10
                n_occupied[i] = True
                # break

        # otherA = np.array(state[10:12])
        # otherB = np.array(state[12:14])
        # dist = np.sqrt(otherA[0] ** 2 + otherA[1] ** 2)
        # if dist < 3.1:  agent_reward -= 0.25
        # dist = np.sqrt(otherB[0] ** 2 + otherB[1] ** 2)
        # if dist < 3.1:  agent_reward -= 0.25

        rew.append(agent_reward)

    return np.array(rew), sum(n_occupied)
