import torch.nn as nn
import torch.nn.functional as F


class ActorCriticPolicy(nn.Module):
    def __init__(self, feature_net, n_actions, n_hidden=1024):
        super(ActorCriticPolicy, self).__init__()
        self.feature_net = feature_net
        self.n_actions = n_actions

        self.actor = nn.Linear(self.feature_net.out_dim, n_hidden)
        self.actor_head = nn.Linear(n_hidden, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        
        self.critic = nn.Linear(self.feature_net.out_dim, n_hidden)
        self.critic_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.feature_net.forward(x)
        return self.softmax(self.actor_head(F.relu(self.actor(x)))), self.critic_head(F.relu(self.critic(x)))
