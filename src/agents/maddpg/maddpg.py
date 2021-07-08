import torch
import torch.nn as nn

from src.agents.maddpg.ddpgagent import DDPGAgent
from src.utils.agent_utils import update_target_network
from src.utils.replaymemory import ReplayMemory
from collections import namedtuple

Transition = namedtuple('maddpg', ('state', 'action', 'reward', 'next_state', 'terminated'))

MSELoss = torch.nn.MSELoss()


class MADDPG(nn.Module):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, state_dim, action_dim, n_ag, gamma=0.95, target_tau=0.01, lr=0.01, hidden_dim=64, batch_size=20,
                 memory_len=50000, name='maddpg', **kwargs):
        super(MADDPG, self).__init__()
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = n_ag
        self.agents = nn.ModuleList(
            [DDPGAgent(state_dim, action_dim, n_ag, lr=lr, hidden_dim=hidden_dim) for _ in range(n_ag)])
        self.gamma = gamma
        self.tau = target_tau
        self.lr = lr
        self.niter = 0

        self.memory = ReplayMemory(memory_len)
        self.memory.transition = Transition
        self.batch_size = batch_size

        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def get_action(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.get_action(obs, explore=explore, device=self.device) for a, obs in zip(self.agents, observations)]

    def fit(self, e):
        sample = self.memory.sample(self.batch_size)

        obs = []
        acs = []
        rews = []
        next_obs = []
        dones = []

        lsted_sample = [obs, acs, rews, next_obs, dones]
        for s in sample:
            for l, ss in zip(lsted_sample, s):
                l.append(ss)

        lsted_sample = [torch.Tensor(l).to(self.device) for l in lsted_sample]

        critic_loss = 0
        actor_loss = 0

        for i in range(self.nagents):
            c, a = self.fit_individual_ag(lsted_sample, i)
            critic_loss += c
            actor_loss += a

        self.update_all_targets()

        return {'critic_loss': critic_loss,
                'actor_loss': actor_loss}

    def fit_individual_ag(self, sample, agent_i):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            all_trgt_acs = [pi(next_obs[:, i, :]) for i, pi in enumerate(self.target_policies)]
            trgt_vf_in = torch.cat((next_obs.reshape(self.batch_size, -1), *all_trgt_acs), dim=1)
            target_value = (rews[:, agent_i].view(-1, 1) + self.gamma * curr_agent.target_critic(trgt_vf_in) * (
                    1 - dones[:, agent_i].view(-1, 1))).view(-1)
        vf_in = torch.cat((obs.reshape(self.batch_size, -1), acs.reshape(self.batch_size, -1)), dim=1)
        actual_value = curr_agent.critic(vf_in).view(-1)
        vf_loss = MSELoss(actual_value, target_value)
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()
        curr_pol_out = curr_agent.policy(obs[:, agent_i, :])
        curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        for i, pi in zip(range(self.nagents), self.policies):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(pi(obs[:, i, :]))

        vf_in = torch.cat((obs.reshape(self.batch_size, -1), *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        return vf_loss.item(), pol_loss.item()

    def update_all_targets(self):
        for a in self.agents:
            update_target_network(a.target_critic.parameters(), a.critic.parameters(), self.tau)
            update_target_network(a.target_policy.parameters(), a.policy.parameters(), self.tau)
        self.niter += 1

    def push(self, *args):
        self.memory.push(*args)

    def save(self, dirname, e):
        torch.save(self.state_dict(), dirname + "/{}.th".format(e))
