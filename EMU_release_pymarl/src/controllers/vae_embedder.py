import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, args, state_dim ):
        super(VAE, self).__init__()

        self.args      = args
        self.lambda_s  = args.lambda_s
        self.lambda_kl = args.lambda_kl

        self.latent_dim    = args.emdqn_latent_dim
        self.hidden_dim    = 64
        self.state_dim     = state_dim
        self.condition_dim = 1
        self.H_dim         = 1
        
        self.input_dim = self.state_dim + self.condition_dim

        if self.args.encoder_type == 1: # FC encoder
            self.encoder = Encoder(args, self.input_dim, self.hidden_dim, self.latent_dim)
        elif self.args.encoder_type == 2: # VAE encoder
            self.encoder = Encoder_VAE(args, self.input_dim, self.hidden_dim, self.latent_dim)

        self.decoder = Decoder(args, self.latent_dim, self.condition_dim, self.hidden_dim, self.state_dim, self.H_dim )
        self.criterion = nn.MSELoss(reduction="sum") # for reconstruction loss

    def reparameterize(self, mu, log_var, flagTraining):
        #if (self.stochastic_sample == True):
        if flagTraining == True:
            std = th.exp(0.5 * log_var)
            eps = th.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z

    #.. VAE forward --------------------------------------------
    def forward(self, s, y ):
        # inputs: cat[s, y]        
        inputs = th.cat([s, y], dim=2)

        if self.args.encoder_type == 1:
            x = self.encoder( inputs ) 
            decoder_inputs = th.cat([x, y], dim=2)
            s_hat, H_hat = self.decoder( decoder_inputs )

            return s_hat, H_hat

        elif self.args.encoder_type == 2:
            mu, log_var = self.encoder( inputs )
            x = self.reparameterize(mu, log_var, flagTraining=True)
            decoder_inputs = th.cat([x, y], dim=2)
            s_hat, H_hat = self.decoder( decoder_inputs )

            return s_hat, H_hat, mu, log_var
    #-----------------------------------------------------------

    def loss_function_fc(self, s_hat, H_hat, s, H):
        #recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        recon_loss_s = self.criterion(s_hat, s)        
        recon_loss_H = self.criterion(H_hat, H)

        return recon_loss_H + self.lambda_s*recon_loss_s 

    # Define the loss function (negative ELBO) and move it to the GPU
    def loss_function_vae(self, s_hat, H_hat, s, H, mu, log_var):
        # def loss_function_vae(recon_x, x, mu, log_var):
        # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        recon_loss_s = self.criterion(s_hat, s)
        recon_loss_H = self.criterion(H_hat, H)

        kl_divergence = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss_H + self.lambda_s*recon_loss_s + self.lambda_kl*kl_divergence

# Define the encoder architecture
class Encoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = args.device

        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, latent_dim).to(device)        

    def forward(self, inputs ):
        # inputs: cat[s, y]
        #inputs = th.cat([s, y], dim=2)
        bs    = inputs.size()[0]            
        max_t = inputs.size()[1]  

        net_inputs = inputs.reshape(bs * max_t, -1)

        x = F.relu(self.fc1(net_inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.view(bs, max_t, -1)

class Encoder_VAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = args.device

        super(Encoder_VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc31 = nn.Linear(hidden_dim, latent_dim).to(device)
        self.fc32 = nn.Linear(hidden_dim, latent_dim).to(device)

    def forward(self, inputs ):
        # inputs: cat[s, y]
        #inputs = th.cat([s, y], dim=2)
        bs    = inputs.size()[0]            
        max_t = inputs.size()[1]  

        net_inputs = inputs.reshape(bs * max_t, -1)

        x = F.relu(self.fc1(net_inputs))
        x = F.relu(self.fc2(x))
        mu = self.fc31(x)
        log_var = self.fc32(x)
        
        return mu.view(bs, max_t, -1), log_var.view(bs, max_t, -1)

# Define the decoder architecture
class Decoder(nn.Module):
    def __init__(self, args, latent_dim, condition_dim, hidden_dim, output_dim, H_dim):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = args.device
        input_dim = latent_dim + condition_dim

        super(Decoder, self).__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc31 = nn.Linear(hidden_dim, output_dim).to(device)
        self.fc32 = nn.Linear(hidden_dim, H_dim).to(device)

    def forward(self, inputs):
        # inputs: cat[x,y]
        #inputs = th.cat([x, y], dim=2)
        bs    = inputs.size()[0]            
        max_t = inputs.size()[1]  

        net_inputs = inputs.reshape(bs * max_t, -1)

        x = F.relu(self.fc1(net_inputs))
        x = F.relu(self.fc2(x))
        s_hat = self.fc31(x)
        H_hat = self.fc32(x)
        
        return s_hat.view(bs, max_t, -1), H_hat.view(bs, max_t, -1)

#.. obsolete ----------------------------------------------------
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
