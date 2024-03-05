import numpy as np
import torch as th
from torch.optim import RMSprop
from torch.optim import Adam
from modules.agents.LRN_KNN import LRU_KNN
from modules.agents.LRN_KNN_STATE import LRU_KNN_STATE
#from modules.agents.state_embedder import StateEmbedder
from controllers import REGISTRY as mac_REGISTRY

class Episodic_memory_buffer():
    def __init__(self, args, scheme ):

        self.rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        self.use_AEM = args.use_AEM

        self.memory_emb_type  = int(args.memory_emb_type)
        self.random_projection = self.rng.normal(loc=0, scale=1. / np.sqrt(args.emdqn_latent_dim),
                                       size=(args.emdqn_latent_dim, scheme['state']['vshape']))

        self.state_dim = scheme['state']['vshape']

        if self.use_AEM == True: 
            if self.memory_emb_type == 2: # use embedding function            
                self.state_embed_net = mac_REGISTRY["State_Embedder"](args,  self.state_dim)
                self.predict_mac     = mac_REGISTRY["V_predictor"](args)
            
                self.predict_params    = list(self.predict_mac.parameters())
                self.predict_params   += list(self.state_embed_net.parameters())

                self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)

                self.ec_buffer = LRU_KNN_STATE(args.emdqn_buffer_size, self.state_dim, args, 'game', self.random_projection, self.state_embed_net)

            elif self.memory_emb_type == 3: # use embedding function with reconstruction loss
                self.VAE             = mac_REGISTRY["VAE"](args, self.state_dim)
                self.state_embed_net = self.VAE.encoder
                self.predict_mac     = self.VAE.decoder
                self.reparameterize  = self.VAE.reparameterize

                self.predict_params    = list(self.predict_mac.parameters())
                self.predict_params   += list(self.state_embed_net.parameters())

                self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)

                self.ec_buffer = LRU_KNN_STATE(args.emdqn_buffer_size, self.state_dim, args, 'game', self.random_projection, self.state_embed_net)

            else: # use random projection
                self.ec_buffer = LRU_KNN_STATE(args.emdqn_buffer_size, self.state_dim, args, 'game', self.random_projection )
        else:
            self.ec_buffer = LRU_KNN(args.emdqn_buffer_size, args.emdqn_latent_dim, 'game')

        self.ec_buffer.strategy = np.zeros([args.emdqn_buffer_size, args.n_agents])
        
        self.q_episodic_memeory_cwatch = []
        self.args=args
        self.update_counter =0
        self.qecwatch=[]
        self.qec_found=0        
        self.update_counter_call =0
        self.is_update_required = False
        self.device = args.device
        self.prediction_loss_monitor = th.zeros(1)

        #.. training part
        #self.batch_size = int(1024) # batch size per training
        #self.mini_batch_size = int(128)
        self.batch_size = int(args.emb_training_batch) # batch size per training
        self.mini_batch_size = int(args.emb_training_mini_batch)
        self.n_epoch = int(self.batch_size / self.mini_batch_size)
        #self.n_update = int(self.ec_buffer.capacity / self.batch_size) # 1M/1000=1000
        
    def update_kdtree(self):
        self.ec_buffer.update_kdtree()
        
    #.. updated version ----------------------------------------------------------------------------------------------------------
    def can_sample(self, batch_size):
        return self.ec_buffer.curr_capacity >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.ec_buffer.curr_capacity == batch_size:
            #return self[:batch_size]
            return np.arange(0, batch_size)
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.ec_buffer.curr_capacity, batch_size, replace=False) # array([x,x,x,...]).int32
            return ep_ids

    def update_ec_buffer_stats(self):
        self.ec_buffer.update_states_norm()

    def peek_modified(self, key, value_decay, xi, modify, global_state, cur_time):        
        return self.ec_buffer.peek_modified_EC(key, value_decay, xi, modify, global_state, cur_time)

    def peek_EC(self, key, value_decay, modify):
        return self.ec_buffer.peek_EC(key, value_decay, modify)
      
    #---------------------------------------------------------------------------------------------------------------------------------------    
    def update_ec_modified(self, episode_batch): 
        ep_state = episode_batch['state'][0, :] # [time, states=140]
        ep_action = episode_batch['actions'][0, :] # [time, agents, 1]
        ep_reward = episode_batch['reward'][0, :] # [time, 1]
        ep_winflag  = episode_batch['flag_win'][0, :] # [time, 1] 

        Rtd = 0.
        xi_tau  = 0
        flag_start = False
        te = episode_batch.max_seq_length
 
        for t in range(episode_batch.max_seq_length - 1, -1, -1):
            s = ep_state[t]
            a = ep_action[t]
            r = ep_reward[t]            

            if xi_tau == 0:
                xi_tau= ep_winflag[t] # check current time and once it becomes 1 then xi always 1 for that episode

            Rtd = r + self.args.gamma * Rtd

            if (sum(s)!=0) and (flag_start==False):                
                flag_start = True
                if self.memory_emb_type == 2: # state embedding
                    te = t
                    state_input = (ep_state[0:te+1]) # (time, s_dim)
                    state_input_exp = state_input.unsqueeze(0) # (bs, time, s_dim)
                    z_input = (self.state_embed_net(state_input_exp)).squeeze(0) # (time, s_dim)

                elif self.memory_emb_type == 3: # state embedding with cVAE
                    te = t
                    state_input = (ep_state[0:te+1]) # (time, s_dim)
                    time_input  = (th.arange(te+1)/float(episode_batch.max_seq_length)).to(self.device).unsqueeze(1) # (time, 1)
                    state_input_exp = state_input.unsqueeze(0) # (bs=1, time, s_dim)
                    time_input_exp  = time_input.unsqueeze(0).float() # (bs=1, time, 1)
                    embed_input     = th.cat([state_input_exp, time_input_exp], dim=2)
                    if self.args.encoder_type == 1:
                        z_input = (self.state_embed_net( embed_input )).squeeze(0) # (time, z_dim)
                    elif self.args.encoder_type == 2:
                        mu, log_var = (self.state_embed_net( embed_input )) 
                        z_input = self.reparameterize(mu, log_var, flagTraining=False).squeeze(0) # (time, z_dim)

            if flag_start==True: # start obtaining valid inputs            
                s_in = s.flatten().detach().cpu().numpy()

                if self.memory_emb_type == 1:
                    z = np.dot(self.random_projection, s.flatten().cpu()) # [emdqn_dim=4]
                    z = z.reshape((self.args.emdqn_latent_dim)) 
                elif self.memory_emb_type == 2:
                    z    = z_input[t].flatten().detach().cpu().numpy()                    
                elif self.memory_emb_type == 3:
                    z    = z_input[t].flatten().detach().cpu().numpy()                    

                qd, xi_t, dummy = self.ec_buffer.peek_modified_EC(z, Rtd, xi_tau, True, s_in, t) # input: z (cpu)
            
                if xi_t == 1: # optimality propagation
                    xi_tau = 1
            
                if (qd == None) :
                    self.ec_buffer.add_modified_EC(z, Rtd, xi_tau, s_in, t)                    

    def update_ec_original(self, episode_batch): 
        ep_state = episode_batch['state'][0, :] # [time, states=140]
        ep_action = episode_batch['actions'][0, :] # [time, agents, 1]
        ep_reward = episode_batch['reward'][0, :] # [time, 1]        
        Rtd = 0.
        for t in range(episode_batch.max_seq_length - 1, -1, -1):
            s = ep_state[t]
            a = ep_action[t]
            r = ep_reward[t]            
            z = np.dot(self.random_projection, s.flatten().cpu()) # [emdqn_dim=4]
            Rtd = r + self.args.gamma * Rtd
            z = z.reshape((self.args.emdqn_latent_dim)) 
            qd = self.ec_buffer.peek_EC(z, Rtd, True)
            #qd, _ = self.ec_buffer.peek(z, Rtd, True)

            if (qd == None) and (sum(s)!=0):  # new action
                self.ec_buffer.add_EC(z, Rtd)

    def hit_probability(self):
        return (1.0 * self.qec_found / self.args.batch_size / self.update_counter)

    def check_tolerance(self):
        return self.ec_buffer.check_tolerance()

    def check_Ncall(self):
        return self.ec_buffer.check_Ncall()

    def update_embedding(self):        
        if self.is_update_required== True:
            n_update = int(self.ec_buffer.curr_capacity / self.batch_size ) + 1
            for iter in range(0, n_update):
                ids = self.batch_size*(iter)
                if iter == n_update - 1:
                    ide = self.ec_buffer.curr_capacity
                else:
                    ide = self.batch_size*(iter+1)                    

                if self.memory_emb_type == 2: # FC
                    if ids==ide:            
                        batch_state = th.tensor(self.ec_buffer.global_states[ids,:]).unsqueeze(0).to(self.device)  # [1,dim] numpy --> torch
                    else:
                        batch_state = th.tensor(self.ec_buffer.global_states[ids:ide,:]).to(self.device)  # [bs,dim], numpy --> torch
                    embed_state = self.state_embed_net( batch_state.unsqueeze(1) ).squeeze(1)             # [bs,1,dim]

                elif self.memory_emb_type == 3: # cVAE
                    if ids==ide:            
                        batch_state = th.tensor(self.ec_buffer.global_states[ids,:]).unsqueeze(0).to(self.device)  # [1,dim] numpy --> torch
                        batch_time  = th.tensor(self.ec_buffer.tg[ids]).unsqueeze(0).to(self.device) / float(self.args.episode_limit)
                    else:
                        batch_state = th.tensor(self.ec_buffer.global_states[ids:ide,:]).to(self.device)  # [bs,dim], numpy --> torch
                        batch_time  = th.tensor(self.ec_buffer.tg[ids:ide]).to(self.device) / float(self.args.episode_limit)

                    embed_input = th.cat( [ batch_state, batch_time.unsqueeze(1)] , dim=1 ).unsqueeze(1)

                    if self.args.encoder_type == 1:
                        embed_state = self.state_embed_net( embed_input ).squeeze(1)             # [bs,dim]
                    elif self.args.encoder_type == 2:
                        mu, log_var = (self.state_embed_net( embed_input )) 
                        embed_state = self.reparameterize(mu, log_var, flagTraining=False).squeeze(1) # (time, z_dim)

                #..update embed_state with updated ones
                self.ec_buffer.states[ids:ide,:] = embed_state.detach().cpu().numpy()  # torch--> numpy

            #.. update is done
            self.is_update_required == False

        return

    def train_embedder(self):
        if self.ec_buffer.build_tree == False :
            return

        if self.can_sample(self.batch_size):
            ep_idx = self.sample( self.batch_size )
            
            batch_state = th.tensor(self.ec_buffer.global_states[ep_idx,:]).to(self.device)  # numpy --> torch
            batch_H     = th.tensor(self.ec_buffer.q_values_decay[ep_idx]).to(self.device)   # numpy --> torch
            batch_time  = th.tensor(self.ec_buffer.tg[ep_idx]).to(self.device)               # numpy --> torch

            #.. slice into minibatch size and us them for learning
            for iter in range(0, self.n_epoch):
                # step1. minibatch sampling (s, H) from EM
                ids = self.mini_batch_size*(iter)
                ide = self.mini_batch_size*(iter+1)

                # step2. supervised training setting (s, H) from EM
                if self.memory_emb_type == 2: # FC
                    embed_state = self.state_embed_net( batch_state[ids:ide].unsqueeze(1) ) # [bs,1,dim]
                    Hest        = (self.predict_mac( embed_state )).squeeze(1) # [bs,1]
                    Hout        = batch_H[ids:ide].detach() # [bs,1]

                    #prediction_loss = th.nn.MSELoss( Hout , Hest ) # mean
                    prediction_loss = (( Hout - Hest )**2).mean() # mean

                elif self.memory_emb_type == 3: # cVAE

                    state_input = batch_state[ids:ide].unsqueeze(1) # [bs,1,dim]
                    time_input  = batch_time[ids:ide].unsqueeze(1).unsqueeze(2) / float(self.args.episode_limit) # [bs,1,dim]
                    #embed_input = th.cat( [ batch_state[ids:ide], batch_time[ids:ide] ] ).unsqueeze(1)
                    #embed_state = self.state_embed_net( batch_state[ids:ide].unsqueeze(1) ) # [bs,1,dim]                    
                    #Hest        = (self.predict_mac( embed_state )).squeeze(1) # [bs,1]
                    Hout        = batch_H[ids:ide].unsqueeze(1).unsqueeze(2).detach() # [bs,1,1]
                    state_out   = batch_state[ids:ide].unsqueeze(1).detach()

                    if self.args.encoder_type == 1: # FC encoder
                        state_est, Hest = self.VAE( state_input, time_input )
                        prediction_loss = self.VAE.loss_function_fc( state_est, Hest, state_out, Hout )
                    elif self.args.encoder_type == 2: # cVAE encoder
                        state_est, Hest, mu, log_var = self.VAE( state_input, time_input )
                        prediction_loss = self.VAE.loss_function_vae( state_est, Hest, state_out, Hout, mu, log_var )

                    #prediction_loss = th.nn.MSELoss( Hout , Hest ) # mean
                    #prediction_loss = (( Hout - Hest )**2).mean() # mean
                    
                self.prediction_loss_monitor = prediction_loss.detach()

                self.predict_optimiser.zero_grad()
                prediction_loss.backward()
                predict_grad_norm = th.nn.utils.clip_grad_norm_(self.predict_params, self.args.grad_norm_clip)
                self.predict_optimiser.step()

            self.is_update_required = True
        #return (1)