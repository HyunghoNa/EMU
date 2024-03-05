import copy
import os
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from modules.mixers.qmix_central_attention import QMixerCentralAtten
import torch as th
from torch.optim import RMSprop
from utils.torch_utils import to_cuda
from torch.optim import Adam
import torch.nn.functional as func
from controllers import REGISTRY as mac_REGISTRY
import numpy as np
from .vdn_Qlearner import vdn_QLearner

class fast_QLearner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        #.. EMU ----------------------------------
        self.use_AEM         = args.use_AEM
        self.memory_emb_type = args.memory_emb_type              
        #-----------------------------------------

        self.last_target_update_episode = 0
        self.save_buffer_cnt = 0
        if self.args.save_buffer:
            self.args.save_buffer_path = os.path.join(self.args.save_buffer_path, str(self.args.seed))

        self.mixer = None

        ###curiosity new
    
        self.vdn_learner=vdn_QLearner(mac, scheme, logger, args, groups=groups)
        self.decay_stats_t = 0
        self.state_shape = scheme["state"]["vshape"]

        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
                self.soft_update_target_mixer = copy.deepcopy(self.mixer)
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
                self.soft_update_target_mixer = copy.deepcopy(self.mixer)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.save_buffer_cnt = 0

        self.n_actions = self.args.n_actions

    def subtrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, intrinsic_rewards,
                 ec_buffer=None, save_buffer=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        #actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True) # [batch, time, agent, actions]

        if save_buffer:
            curiosity_r=intrinsic_rewards.clone().detach().cpu().numpy()
            # rnd_r = rnd_intrinsic_rewards.clone().detach().cpu().numpy()
            # extrinsic_mac_out_save=extrinsic_mac_out.clone().detach().cpu().numpy()
            mac_out_save = mac_out.clone().detach().cpu().numpy()
            actions_save=actions.clone().detach().cpu().numpy()
            terminated_save=terminated.clone().detach().cpu().numpy()
            state_save=batch["state"][:, :-1].clone().detach().cpu().numpy()
            data_dic={'curiosity_r':curiosity_r,
                                 # 'extrinsic_Q':extrinsic_mac_out_save,
                        'control_Q':mac_out_save,'actions':actions_save,'terminated':terminated_save,
                        'state':state_save}

            self.save_buffer_cnt += self.args.save_buffer_cycle

            if not os.path.exists(self.args.save_buffer_path):
                os.makedirs(self.args.save_buffer_path)
            np.save(self.args.save_buffer_path +"/"+ 'data_{}'.format(self.save_buffer_cnt), data_dic)
            print('save buffer ({}) at time{}'.format(batch.batch_size, self.save_buffer_cnt))
            return

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()



        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        if self.args.use_emdqn:
            ec_buffer.update_counter += 1
            qec_input = chosen_action_qvals.clone().detach()
            qec_input_new = []
            eta  = th.zeros_like(qec_input).detach().to(self.args.device)

            if self.use_AEM == False: # EMC
                for i in range(self.args.batch_size): # batch = 32
                    qec_tmp = qec_input[i, :]
                    for j in range(1, batch.max_seq_length):
                        if not mask[i, j - 1]:
                            continue           
                        ec_buffer.update_counter_call += 1 
                        z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())
                        q = ec_buffer.peek_EC(z, None, modify=False)
                        if q != None:
                            qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1]
                            ec_buffer.qecwatch.append(q)
                            ec_buffer.qec_found += 1
                    qec_input_new.append(qec_tmp)
                qec_input_new = th.stack(qec_input_new, dim=0)

            else: # EMU
                Vopt = target_max_qvals.clone().detach() # default value                

                for i in range(self.args.batch_size): # batch = 32
                    qec_tmp = qec_input[i, :]
                    for j in range(1, batch.max_seq_length):
                        if not mask[i, j - 1]:
                            continue                 

                        ec_buffer.update_counter_call += 1 
                        if self.memory_emb_type == 1:
                            z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())
                        elif self.memory_emb_type == 2:
                            z = ec_buffer.state_embed_net(batch["state"][i][j].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy() # can be improved via batch-run
                        elif self.memory_emb_type == 3:
                            timestep = th.tensor( [float(j) / float(self.args.episode_limit)] ).to(self.args.device)
                            embed_input = th.cat( [ batch["state"][i][j], timestep], dim=0).unsqueeze(0).unsqueeze(0)

                            if self.args.encoder_type == 1: # FC
                                z = ec_buffer.state_embed_net( embed_input ).squeeze(0).squeeze(0).detach().cpu().numpy() # can be improved via batch-run
                            elif self.args.encoder_type == 2: # cVAE
                                mu, log_var = ec_buffer.state_embed_net( embed_input ) # can be improved via batch-run 
                                z = ec_buffer.reparameterize(mu, log_var, flagTraining=False).squeeze(0).squeeze(0).detach().cpu().numpy()

                        q, xi, rcnt = ec_buffer.peek_modified(z, None, 0, modify=False, global_state=None, cur_time=0)
                        
                        if q != None:
                            qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1]
                            ec_buffer.qecwatch.append(q)
                            ec_buffer.qec_found += 1
                            Vopt[i][j-1][0] = th.tensor(q).to(self.args.device)

                        if self.args.optimality_type == 1 and rcnt !=None: # expected
                            eta[i][j-1] = rcnt * max(Vopt[i][j-1] - target_max_qvals[i][j-1], 0.0)
                        elif self.args.optimality_type == 2 and xi != None : # optimistic
                            eta[i][j-1] = xi * max(Vopt[i][j-1] - target_max_qvals[i][j-1], 0.0)
                        
                    qec_input_new.append(qec_tmp)
                qec_input_new = th.stack(qec_input_new, dim=0)
            

            # print("qec_mean:", np.mean(ec_buffer.qecwatch))
            # episodic_q_hit_pro = 1.0 * ec_buffer.qec_found / self.args.batch_size / ec_buffer.update_counter / batch.max_seq_length
            # episodic_qec_hit_pro_norm =  ec_buffer.qec_found /  ec_buffer.update_counter_call
            # print("qec_fount: %.2f" % episodic_q_hit_pro)
        
        #targets = float(self.args.optimality_incentive)*self.args.gamma*eta + intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        

        if self.args.optimality_incentive:
            targets = self.args.gamma*eta + intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        
        else:
            targets = intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)
        if self.args.use_emdqn:
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals          
            #emdqn_masked_td_error = emdqn_td_error * mask * (1-float(self.args.optimality_incentive))
            if self.args.optimality_incentive:
                emdqn_masked_td_error = emdqn_td_error * mask * 0.0
            else:
                emdqn_masked_td_error = emdqn_td_error * mask

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.use_emdqn:
            emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
            loss += emdqn_loss

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # if self.args.use_emdqn:
            #     self.logger.log_stat("e_m Q mean",  (qec_input_new * mask).sum().item() /
            #                          (mask_elems * self.args.n_agents), t_env)
            #     self.logger.log_stat("em_ Q hit probability", episodic_q_hit_pro, t_env)
            #     self.logger.log_stat("emdqn_loss", emdqn_loss.item(), t_env)
            #     self.logger.log_stat("emdqn_curr_capacity", ec_buffer.ec_buffer.curr_capacity, t_env)
            #     self.logger.log_stat("emdqn_weight", self.args.emdqn_loss_weight, t_env)
            #     self.logger.log_stat("qec_hit_prob_norm", episodic_qec_hit_pro_norm, t_env)
            #     self.logger.log_stat("eta_mean",
            #                      (eta * mask).sum().item() / (mask_elems), t_env)

            self.logger.log_stat("extrinsic rewards", rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                        
            self.log_stats_t = t_env

            # if self.use_AEM and self.args.use_emdqn:
            #     atol, rtol, x_mu, x_sigma, z_mu, z_sigma = ec_buffer.check_tolerance()
            #     mu_Ncall, mu_Nxi, mu_ratio_xi, curr_capacity = ec_buffer.check_Ncall()
            #     prediction_net_loss = ec_buffer.prediction_loss_monitor

            #     self.logger.log_stat("atol", atol.item(), t_env)
            #     self.logger.log_stat("rtol", rtol.item(), t_env)
            
            #     self.logger.log_stat("mu_Ncall", mu_Ncall.item(), t_env)
            #     self.logger.log_stat("mu_Nxi", mu_Nxi.item(), t_env)
            #     self.logger.log_stat("mu_ratio_xi", mu_ratio_xi.item(), t_env)
            #     self.logger.log_stat("curr_capacity", curr_capacity, t_env)
                                
            #     self.logger.log_stat("prediction_net_loss", prediction_net_loss.item(), t_env)

            #     self.logger.log_stat("x1_mu", x_mu[0].item(), t_env)
            #     self.logger.log_stat("x2_mu", x_mu[1].item(), t_env)
            #     self.logger.log_stat("x3_mu", x_mu[2].item(), t_env)
            #     self.logger.log_stat("x4_mu", x_mu[3].item(), t_env)

            #     self.logger.log_stat("x1_sigma", x_sigma[0].item(), t_env)
            #     self.logger.log_stat("x2_sigma", x_sigma[1].item(), t_env)
            #     self.logger.log_stat("x3_sigma", x_sigma[2].item(), t_env)
            #     self.logger.log_stat("x4_sigma", x_sigma[3].item(), t_env)

            #     self.logger.log_stat("z1_mu", z_mu[0].item(), t_env)
            #     self.logger.log_stat("z2_mu", z_mu[1].item(), t_env)
            #     self.logger.log_stat("z3_mu", z_mu[2].item(), t_env)
            #     self.logger.log_stat("z4_mu", z_mu[3].item(), t_env)

            #     self.logger.log_stat("z1_sigma", z_sigma[0].item(), t_env)
            #     self.logger.log_stat("z2_sigma", z_sigma[1].item(), t_env)
            #     self.logger.log_stat("z3_sigma", z_sigma[2].item(), t_env)
            #     self.logger.log_stat("z4_sigma", z_sigma[3].item(), t_env)
                
        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False,
              ec_buffer=None):
        intrinsic_rewards = \
            self.vdn_learner.train(batch, t_env, episode_num, save_buffer=False, imac=self.mac, timac=self.target_mac)
        if self.args.is_prioritized_buffer:
            masked_td_error, mask = self.subtrain(batch, t_env, episode_num, self.mac,
                                                  intrinsic_rewards=intrinsic_rewards,
                                                  ec_buffer=ec_buffer)

        else:
            self.subtrain(batch, t_env, episode_num, self.mac, intrinsic_rewards=intrinsic_rewards, ec_buffer=ec_buffer)

        if hasattr(self.args, 'save_buffer') and self.args.save_buffer:
            if self.buffer.episodes_in_buffer - self.save_buffer_cnt >= self.args.save_buffer_cycle:
                if self.buffer.can_sample(self.args.save_buffer_cycle):
                    batch_tmp=self.buffer.sample(self.args.save_buffer_cycle, newest=True)
                    intrinsic_rewards_tmp, _ = \
                        self.vdn_learner.train(batch_tmp, t_env, episode_num, save_buffer=True,
                                                 imac=self.mac, timac=self.target_mac)
                    self.subtrain(batch_tmp, t_env, episode_num, self.mac, intrinsic_rewards=intrinsic_rewards_tmp,
                                  save_buffer=True)


                else:
                    print('**' * 20, self.buffer.episodes_in_buffer, self.save_buffer_cnt)


        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(ec_buffer)
            self.last_target_update_episode = episode_num
            

        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res


    def _update_targets(self,ec_buffer=None):
        if self.args.use_emdqn:
            ec_buffer.update_kdtree()
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")




    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        self.vdn_learner.cuda()

        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)
            to_cuda(self.soft_update_target_mixer, self.args.device)

    def save_models(self, path, ec_buffer): ## save models from here...
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            th.save(self.target_mixer.state_dict(), "{}/target_mixer.th".format(path))

        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.args.use_emdqn == True and ( self.args.memory_emb_type == 2 or self.args.memory_emb_type == 3 ):
            th.save(ec_buffer.predict_mac.state_dict(), "{}/predict_mac.th".format(path))
            th.save(ec_buffer.state_embed_net.state_dict(), "{}/state_embed_net.th".format(path))

        #.. save model related to episodic memory
        if (ec_buffer is not None) and self.args.save_memory_info:
            if self.use_AEM: 
                ec_buffer.ec_buffer.save_memory(path)

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/target_mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
            self.soft_update_target_mixer.load_state_dict(th.load("{}/soft_update_target_mixer.th".format(path),
                                                                  map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
