import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DMAQ_Simple_Weight(nn.Module):
    def __init__(self, args):
        super(DMAQ_Simple_Weight, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        adv_hypernet_embed = self.args.adv_hypernet_embed
        if getattr(args, "adv_hypernet_layers", 1) == 1:
            self.key = nn.Linear(self.state_action_dim, self.n_agents)  # key
            self.action = nn.Linear(self.state_action_dim, self.n_agents)  # action
        elif getattr(args, "adv_hypernet_layers", 1) == 2:
            self.key = nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                     nn.ReLU(),
                                     nn.Linear(adv_hypernet_embed, self.n_agents))  # key
            self.action = nn.Sequential(nn.Linear(self.state_action_dim, adv_hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(adv_hypernet_embed, self.n_agents))  # action

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = th.cat([states, actions], dim=1)

        x_key = th.abs(self.key(data)) + 1e-10
        x_agents = F.sigmoid(self.action(data))
        weights = x_key * x_agents
        return weights
