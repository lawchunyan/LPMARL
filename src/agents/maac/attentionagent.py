from torch.optim import Adam
from src.utils.agent_utils import update_target_network
from src.agents.maac.policy import DiscretePolicy


class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """

    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64, lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol, hidden_dim=hidden_dim, onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol, num_out_pol, hidden_dim=hidden_dim, onehot_dim=onehot_dim)

        update_target_network(self.target_policy.parameters(), self.policy.parameters())
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, explore=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
