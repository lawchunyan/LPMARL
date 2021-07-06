import torch
import torch.nn as nn

from torch.optim import Adam

from src.nn.MultiLayeredPerceptron import MultiLayeredPerceptron as MLP
from src.utils.agent_utils import update_target_network
from src.utils.OUNoise import OUNoise


class DDPGAgent(nn.Module):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

    def __init__(self, state_dim, action_dim, n_ag, hidden_dim=64, lr=0.01):
        super(DDPGAgent, self).__init__()
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLP(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim], out_activation=nn.Tanh())
        self.target_policy = MLP(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim], out_activation=nn.Tanh())

        self.critic = MLP((state_dim + action_dim) * n_ag, 1, hidden_dims=[hidden_dim, hidden_dim],
                          out_activation=nn.Identity())
        self.target_critic = MLP((state_dim + action_dim) * n_ag, 1, hidden_dims=[hidden_dim, hidden_dim],
                                 out_activation=nn.Identity())

        update_target_network(self.target_critic.parameters(), self.critic.parameters())
        update_target_network(self.target_critic.parameters(), self.critic.parameters())

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.noise = OUNoise(action_dim)

    def reset_noise(self):
        self.noise.reset()

    def get_action(self, obs, explore=False, device='cpu'):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        obs = torch.Tensor(obs).to(device)
        action = self.policy(obs)
        action = action.cpu().detach().numpy()

        if explore:
            action = self.noise.get_action(action)

        return action


