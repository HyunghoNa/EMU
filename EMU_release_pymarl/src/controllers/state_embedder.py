import torch as th
import torch.nn as nn
import torch.nn.functional as F

class StateEmbedder(nn.Module):
    #def __init__(self, scheme, args):
    def __init__(self, args, state_dim):
        super(StateEmbedder, self).__init__()

        self.args = args
        self.state_dim = state_dim

        self.n_actions = args.n_actions
        self.hidden_dim = 64
        self.n_agents = args.n_agents
        self.emdqn_latent_dim = args.emdqn_latent_dim

        self.emb_out_type = int(args.emb_out_type)

        #input_shape = self._get_input_shape(scheme) # batch, time, emdqn_latent_dim
        input_shape = self.emdqn_latent_dim
        #self.output_type = "v"

        # Set up network layers
        if self.emb_out_type == 1:
            self.state_embed_net = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_dim, self.emdqn_latent_dim )).to(self.args.device)
        elif self.emb_out_type == 2:
            self.state_embed_net = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_dim, self.emdqn_latent_dim ),
                                            nn.Tanh()).to(self.args.device)
        else:
            self.state_embed_net = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_dim, self.emdqn_latent_dim ),
                                            nn.LayerNorm(self.emdqn_latent_dim)).to(self.args.device)
                
    def forward(self, inputs, t=None):
        #net_inputs, bs, max_t = self._build_inputs(inputs, t=t)
        bs    = inputs.size()[0]            
        max_t = inputs.size()[1]  

        net_inputs = inputs.reshape(bs * max_t, -1)

        state_embed = self.state_embed_net(net_inputs)
        #state_embed = self.state_embed_net(inputs)

        return state_embed.view(bs, max_t, -1)
        #return state_embed
