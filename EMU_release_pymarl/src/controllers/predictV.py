import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PredictVCritic(nn.Module):
    #def __init__(self, scheme, args):
    def __init__(self, args):
        super(PredictVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.emdqn_latent_dim = args.emdqn_latent_dim

        #input_shape = self._get_input_shape(scheme) # batch, time, emdqn_latent_dim
        input_shape = self.emdqn_latent_dim
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128).to(self.args.device)
        self.fc2 = nn.Linear(128, 128).to(self.args.device)
        self.fc3 = nn.Linear(128, 1).to(self.args.device)
    
    def forward(self, inputs, t=None):
        #net_inputs, bs, max_t = self._build_inputs(inputs, t=t)
        bs = inputs.size()[0]
        max_t = inputs.size()[1]
        
        net_inputs=inputs.reshape(bs * max_t, -1)

        x = F.relu(self.fc1(net_inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        #return q.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        return q.view(bs, max_t, -1)

    #def forward_old(self, inputs, t=None):
    #    #inputs, bs, max_t = self._build_inputs(batch, t=t)

    #    x = F.relu(self.fc1(inputs))
    #    x = F.relu(self.fc2(x))
    #    q = self.fc3(x)
    #    return q
    #def _build_inputs(self, inputs, t=None):
    #    bs = inputs.size[0]
    #    max_t = batch.max_seq_length if t is None else 1
    #    ts = slice(None) if t is None else slice(t, t+1)
    #    inputs = []
    #    # state
    #    inputs.append(batch["state"][:, ts])

    #    # observations
    #    inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

    #    # last actions
    #    if t == 0:
    #        inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
    #    elif isinstance(t, int):
    #        inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
    #    else:
    #        last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
    #        last_actions = last_actions.view(bs, max_t, 1, -1)
    #        inputs.append(last_actions)

    #    inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
    #    return inputs, bs, max_t


    #def _build_inputs_old(self, batch, t=None):
    #    bs = batch.batch_size
    #    max_t = batch.max_seq_length if t is None else 1
    #    ts = slice(None) if t is None else slice(t, t+1)
    #    inputs = []
    #    # state
    #    inputs.append(batch["state"][:, ts])

    #    # observations
    #    inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

    #    # last actions
    #    if t == 0:
    #        inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
    #    elif isinstance(t, int):
    #        inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
    #    else:
    #        last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
    #        last_actions = last_actions.view(bs, max_t, 1, -1)
    #        inputs.append(last_actions)

    #    inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
    #    return inputs, bs, max_t

    #def _get_input_shape(self, scheme):
    #    # state
    #    input_shape = scheme["state"]["vshape"]
    #    # observations
    #    input_shape += scheme["obs"]["vshape"] * self.n_agents
    #    # last actions
    #    input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
    #    return input_shape