import numpy as np
import torch
from torch.distributions import Categorical

from env.util import stack_frames
from util import np_var, var


class A2CAgent:
    def __init__(self, policy, optim, cuda=False, gamma=0.99, n_timesteps=1024):

        self.cuda = cuda
        self.optim = optim
        self.policy = policy
        self.gamma = gamma
        self.n_timesteps = n_timesteps

        if cuda:
            self.policy.cuda()

        self._reset_timesteps()

    def _reset_timesteps(self):
        self.logps = np.zeros(self.n_timesteps)
        self.rewards = np.zeros(self.n_timesteps)
        self.values = np.zeros(self.n_timesteps)
        self.dones = np.zeros(self.n_timesteps)

    def _discount(self):
        summed_reward = 0
        returns = np.zeros_like(self.rewards)

        for i in reversed(range(len(self.rewards))):
            summed_reward = self.rewards[i] + self.gamma * (1 - self.dones[i]) * summed_reward
            returns[i] = summed_reward

        return returns

    def act(self, t, state):
        probs, value = self.policy(var(stack_frames([state]), self.cuda))
        self.values[t] = value

        m = Categorical(probs)
        action = m.sample()

        self.logps[t] = -m.log_prob(action)
        return action.data[0]

    def post_act(self, t, reward, is_terminal, next_state):
        self.rewards[t] = reward
        self.dones[t] = is_terminal

    def epoch_finished(self):
        self.optim.zero_grad()

        returns = self._discount()
        advs = returns - self.values.data[:, 0]

        (self.logps * advs).backward(retain_graph=True)
        (torch.nn.SmoothL1Loss()(self.values[:, 0], np_var(returns, self.cuda))).backward()

        self.optim.step()
        self._reset_timesteps()
        return 0

    def state(self):
        return self.policy.state_dict()

    def load(self, state):
        self.policy.load_state_dict(state)
