import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, feature_net, n_actions, n_hidden=1024, dueling=False):
        super(QNetwork, self).__init__()
        self.feature_net = feature_net
        self.dueling = dueling
        self.n_actions = n_actions

        self.value = nn.Linear(self.feature_net.out_dim, n_hidden)
        self.value_head = nn.Linear(n_hidden, 1 if dueling else n_actions)

        if dueling:
            self.advantage = nn.Linear(self.feature_net.out_dim, n_hidden)
            self.advantage_head = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.feature_net.forward(x)
        value = self.value_head(F.relu(self.value(x)))

        if self.dueling:
            advantage = self.advantage_head(F.relu(self.advantage(x)))
            return (advantage - advantage.mean(dim=-1).unsqueeze(-1)) + value
        else:
            return value
