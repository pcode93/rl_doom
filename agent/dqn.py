from random import random, randint

import numpy as np
import torch

from env.util import stack_frames
from util import np_var, var


class DQNAgent:
    def __init__(self, q_net, target_net, optim, replay_memory, eps_schedule, init_steps=10000,
                 q_update_interval=1, target_update_interval=1000, cuda=False, ddqn=True, gamma=0.99, batch_size=32):

        self.cuda = cuda
        self.optim = optim
        self.q_net = q_net
        self.target_net = target_net
        self.memory = replay_memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.epoch_losses = []
        self.loss = torch.nn.MSELoss()
        self.ddqn = ddqn
        self.eps_schedule = eps_schedule
        self.last_state = None
        self.last_action = None
        self.t = 0
        self.q_update_interval = q_update_interval
        self.target_update_interval = target_update_interval
        self.init_steps = init_steps

        self._update_target_net()

        if cuda:
            self.q_net.cuda()
            self.target_net.cuda()

    def _update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _optimize(self):
        self.optim.zero_grad()

        states, actions, rewards, terminals, next_states = self.memory.sample(self.batch_size)
        batch_idx = list(range(self.batch_size))

        states = stack_frames(states)
        next_states = stack_frames(next_states)
        actions = np.hstack(actions)

        rewards = np_var(np.vstack(rewards), self.cuda).data
        terminals = [i for i, done in enumerate(terminals) if done]

        if self.ddqn:
            next_state_policy_values = self.q_net(var(next_states, self.cuda))
            next_actions = np.argmax(next_state_policy_values.data.cpu().numpy(), 1)

            next_state_target_values = self.target_net(var(next_states, self.cuda))
            targets = self.gamma * next_state_target_values.data[batch_idx, next_actions]
        else:
            targets = self.gamma * self.target_net(var(next_states)).data.max(1)

        if terminals:
            targets[terminals] = 0

        outputs = self.q_net(var(states, self.cuda))[batch_idx, actions]

        loss = self.loss(outputs, var(rewards[:, 0] + targets, self.cuda))
        loss.backward()

        self.optim.step()
        self.epoch_losses.append(loss.data[0])

    def act(self, t, state):
        self.last_state = state

        if random() < self.eps_schedule.param:
            action = randint(0, self.q_net.n_actions - 1)
        else:
            action = np.argmax(self.q_net(var(stack_frames([state]), self.cuda)).data[0, :].cpu().numpy(), 0)

        self.last_action = action
        return action

    def post_act(self, t, reward, is_terminal, next_state):
        self.memory.add(self.last_state, self.last_action, reward, is_terminal, next_state)

        if self.t > self.init_steps and self.t % self.q_update_interval == 0:
            self._optimize()

        if self.t % self.target_update_interval == 0:
            self._update_target_net()

        self.t += 1

    def epoch_finished(self):
        losses = np.mean(self.epoch_losses) if self.epoch_losses else 0
        self.epoch_losses = []

        return losses

    def state(self):
        return {
            "model": self.q_net.state_dict(),
            "eps": self.eps_schedule
        }

    def load(self, state):
        self.q_net.load_state_dict(state["model"])
        self.eps_schedule = state["eps"]