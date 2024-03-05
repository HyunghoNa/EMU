import torch.nn as nn
import torch.nn.functional as F
import torch as th


class RNN_individualQ_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN_individualQ_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.agent_num = args.n_agents
        self.n_actions=args.n_actions
        self.count=0
        self.individual_Q_layer = nn.ModuleList()
        for i in range(self.agent_num):
            self.individual_Q_layer.append(nn.Linear(args.rnn_hidden_dim, args.n_actions))

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        q = self.fc2(h)


        q_individual=[]
        h_new = h.view(-1, self.args.n_agents,self.args.rnn_hidden_dim)
        for i in range(self.agent_num):
            q_in=self.individual_Q_layer[i](h_new[:,i])
            #q_innew=q_in.view(-1,1,self.args.n_actions)
            q_individual.append(q_in)
        #q_individual=th.cat(q_individual, dim=0)
        q_individual = th.stack(q_individual, dim=1).view(-1,self.args.n_actions)


        return q+q_individual, h, q_individual
