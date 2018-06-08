import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from env.util import stack_frames
from util import np_var, var


class A2CAgent:
    def __init__(self, policy, optim, cuda=False, gamma=0.99, n_timesteps=1024, batch_size=32):

        self.cuda = cuda
        self.optim = optim
        self.policy = policy
        self.gamma = gamma
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size

        if cuda:
            self.policy.cuda()

        self._reset_timesteps()

    def _reset_timesteps(self):
        self.actions = np.zeros(self.n_timesteps)
        self.states = np.zeros(self.n_timesteps, dtype=object)
        self.rewards = np.zeros(self.n_timesteps)
        self.dones = np.zeros(self.n_timesteps)

    def _discount(self):
        summed_reward = 0
        returns = np.zeros_like(self.rewards)

        for i in reversed(range(len(self.rewards))):
            summed_reward = self.rewards[i] + self.gamma * (1 - self.dones[i]) * summed_reward
            returns[i] = summed_reward

        return returns

    def act(self, t, state):
        self.states[t] = state
        probs, _ = self.policy(var(stack_frames([state]), self.cuda))
        #self.values[t] = value

        m = Categorical(probs)
        action = m.sample()
        self.actions[t] = action.data[0]

        #self.logps[t] = -m.log_prob(action)
        return action.data[0]

    def post_act(self, t, reward, is_terminal, next_state):
        self.rewards[t] = reward
        self.dones[t] = is_terminal

    def epoch_finished(self):
        self.optim.zero_grad()

        returns = self._discount()
        idx = np.random.permutation(self.n_timesteps)
        batch_range = list(range(self.batch_size))

        for num_batch in range(self.n_timesteps // self.batch_size):
            b_idx = idx[num_batch * self.batch_size: (num_batch + 1) * self.batch_size]
            b_states = np.copy(self.states[b_idx])
            b_returns = np.copy(returns[b_idx])
            b_actions = np.copy(self.actions[b_idx])

            probs, values = self.policy(var(stack_frames(b_states), self.cuda))
            logps = torch.log(probs[batch_range, b_actions])
            advs = np_var(b_returns - values.data.cpu().numpy(), self.cuda)
            ((logps * advs).mean() + torch.nn.SmoothL1Loss()(values, np_var(b_returns, self.cuda))).backward()

        self.optim.step()
        self._reset_timesteps()
        return 0

    def state(self):
        return self.policy.state_dict()

    def load(self, state):
        self.policy.load_state_dict(state)
