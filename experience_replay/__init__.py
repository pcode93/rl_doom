import numpy as np


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.t = 0

        self.states = np.zeros(size, dtype=object)
        self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.terminals = np.zeros(size)
        self.next_states = np.zeros(size, dtype=object)

    def add(self, state, action, reward, terminal, next_state):
        t = self.t % self.size

        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] = reward
        self.terminals[t] = terminal
        self.next_states[t] = next_state

        self.t += 1

    def sample(self, size):
        idx = np.random.randint(0, self.size, size)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.terminals[idx], self.next_states[idx]
