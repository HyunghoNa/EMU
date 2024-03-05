import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dmaq_qatten_weight import Qatten_Weight


class QTRAN_transformation(nn.Module):
    def __init__(self, args):
        super(QTRAN_transformation, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)

    def forward(self, agent_qs, states, actions=None):
        bs = agent_qs.size(0)

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents)  + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        agent_qs = agent_qs.view(bs, -1, self.n_agents)

        return agent_qs
