import numpy as np


def change_to_one_hot(actions, action_dim=5):
    identity = np.eye(action_dim)
    return identity[actions]

#
# def get_joint_state(ag_state, en_state):
#     np.concatenate(ag)