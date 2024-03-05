import torch.nn as nn
import torch.nn.functional as F


class RND_nn_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(RND_nn_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnd_predict_dim)

    #def init_hidden(self):
        # make hidden states on same device as model
     #   return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        x1 = F.relu(self.fc1(inputs))
        x2=  F.relu(self.fc2(x1))
        q = self.fc3(x2)
        return q
