import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import to_cuda


class SCOAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SCOAgent, self).__init__()
        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.key = nn.Linear(args.rnn_hidden_dim, 1)
        self.agent_weights = nn.Linear(args.rnn_hidden_dim, self.n_agents)
        self.action_weights = nn.Linear(args.rnn_hidden_dim, self.n_agents * self.n_actions)

        self.agent_comm_mask = 1. - to_cuda(th.eye(self.n_agents), self.args.device).unsqueeze(0)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, bs, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        x_key = self.key(h)
        x_key = x_key.view(bs, self.n_agents, 1)
        nor_agent_weights = self.agent_weights(h).view(bs, self.n_agents, self.n_agents)
        nor_agent_weights = F.sigmoid(nor_agent_weights)
        #comm_mask = self.agent_comm_mask.repeat(bs, 1, 1)
        #nor_agent_weights = nor_agent_weights * comm_mask
        nor_action_weights = self.action_weights(h).view(bs, self.n_agents, self.n_agents, self.n_actions)
        nor_action_weights = F.softmax(nor_action_weights, dim=-1)

        x_agent_weights = x_key.repeat(1, 1, self.n_agents) * nor_agent_weights
        x_agent_weights = x_agent_weights.unsqueeze(3).repeat(1, 1, 1, self.n_actions)
        x_action_weights = x_agent_weights * nor_action_weights

        x_q = th.sum(x_action_weights, dim=1)
        q += x_q.view(bs * self.n_agents, self.n_actions)

        return q, h
