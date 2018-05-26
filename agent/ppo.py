import numpy as np
import torch
from torch.distributions import Categorical

from env.util import stack_frames
from util import np_var, var


class PPOAgent:
    def __init__(self, policy, optim, eps_schedule, cuda=False, opt_epochs=4, gamma=0.99, lam=0.95,
                 entropy_coeff=0.01, value_coeff=1.0, n_timesteps=1024, batch_size=32):

        self.cuda = cuda
        self.optim = optim
        self.policy = policy
        self.eps_schedule = eps_schedule
        self.opt_epochs = opt_epochs
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.n_timesteps = n_timesteps
        self.n_batches = n_timesteps // batch_size
        self.batch_size = batch_size

        if cuda:
            self.policy.cuda()

        self._reset_timesteps()

    def _reset_timesteps(self):
        self.old_probs = np.zeros(self.n_timesteps)
        self.actions = np.zeros(self.n_timesteps)
        self.states = np.zeros(self.n_timesteps, dtype=object)
        self.rewards = np.zeros(self.n_timesteps)
        self.values = np.zeros(self.n_timesteps)
        self.dones = np.zeros(self.n_timesteps)
        self.last_state = None

    def _adv(self, rewards, values, dones):
        summed_reward = 0
        returns = [0] * len(rewards)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (1 - dones[i]) * values[i + 1] - values[i]
            summed_reward = summed_reward * (1 - dones[i]) * self.gamma * self.lam + delta
            returns[i] = summed_reward

        return returns

    def act(self, t, state):
        self.states[t] = state
        probs, value = self.policy(var(stack_frames([state]), self.cuda))
        self.values[t] = value.cpu().data[0, 0]

        m = Categorical(probs.cpu().data[0])
        action = m.sample()

        self.actions[t] = action[0]
        self.old_probs[t] = m.log_prob(action)[0]

        return action[0]

    def post_act(self, t, reward, is_terminal, next_state):
        self.rewards[t] = reward
        self.dones[t] = is_terminal
        self.last_state = next_state

    def epoch_finished(self):
        _, value = self.policy(var(stack_frames([self.last_state]), self.cuda))

        advs = np.vstack(self._adv(self.rewards, np.append(self.values, value.cpu().data[0, 0]), self.dones))
        returns = np.vstack(self.values) + np.copy(advs)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        states = np.array(self.states, dtype=object)
        old_probs = np.vstack(self.old_probs)

        batch_range = list(range(self.batch_size))

        for n in range(self.opt_epochs):
            idx = np.random.permutation(self.n_timesteps)

            for num_batch in range(self.n_batches):
                b_idx = idx[num_batch * self.batch_size: (num_batch + 1) * self.batch_size]

                b_advs = np.copy(advs[b_idx])
                b_returns = np.copy(returns[b_idx])
                b_old_probs = np.copy(old_probs[b_idx])
                b_states = stack_frames(states[b_idx])
                b_actions = np.copy(np.array(self.actions)[b_idx])

                self.optim.zero_grad()

                probs, values = self.policy(var(b_states, self.cuda))
                r = torch.exp(torch.log(probs[batch_range, b_actions]) - np_var(b_old_probs, self.cuda).squeeze(1))
                A = np_var(b_advs, self.cuda).squeeze(1)

                L_policy = -torch.min(r * A, r.clamp(1 - self.eps_schedule.param, 1 + self.eps_schedule.param) * A).mean()
                L_value = (values - np_var(b_returns, self.cuda)).pow(2).mean()
                L_entropy = (-((probs * torch.log(probs)).sum(-1))).mean()

                L = self.value_coeff * L_value + L_policy - self.entropy_coeff * L_entropy

                L.backward()
                self.optim.step()

        self._reset_timesteps()
        return L.data.cpu()[0]

    def state(self):
        return self.policy.state_dict()

    def load(self, state):
        self.policy.load_state_dict(state)
