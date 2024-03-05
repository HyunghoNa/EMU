import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from modules.intrinsic.predict_net import Predict_Network1, Predict_Network1_combine
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import RMSprop
from utils.torch_utils import to_cuda
from .vdn_Qlearner import vdn_QLearner
import os


class QPLEX_curiosity_vdn_Learner_cds:
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
    
        self.mixer = None
        #self.vdn_learner = vdn_QLearner(mac, scheme, logger, args)
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            
        # for cds prediction------------------------------------------
        self.eval_predict_withoutid = Predict_Network1(
            args.rnn_hidden_dim + args.obs_shape + args.n_actions, 128, args.obs_shape, False)
        self.target_predict_withoutid = Predict_Network1(
            args.rnn_hidden_dim + args.obs_shape + args.n_actions, 128, args.obs_shape, False)

        self.eval_predict_withid = Predict_Network1_combine(args.rnn_hidden_dim + args.obs_shape + args.n_actions + args.n_agents, 128,
                                                            args.obs_shape, args.n_agents, False)
        self.target_predict_withid = Predict_Network1_combine(args.rnn_hidden_dim + args.obs_shape + args.n_actions + args.n_agents, 128,
                                                              args.obs_shape, args.n_agents, False)

        if self.args.use_cuda:
            self.eval_predict_withid.to(self.args.device)
            self.target_predict_withid.to(self.args.device)

            self.eval_predict_withoutid.to(self.args.device)
            self.target_predict_withoutid.to(self.args.device)

        self.target_predict_withid.load_state_dict(
            self.eval_predict_withid.state_dict())
        self.target_predict_withoutid.load_state_dict(
            self.eval_predict_withoutid.state_dict())
        #-------------------------------------------------------------

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.save_buffer_cnt = 0

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,intrinsic_rewards,
                  show_demo=False, save_data=None, show_v=False, save_buffer=False,ec_buffer=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        # for cds----------------------------------------------------
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions

        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        # for cds----------------------------------------------------
        initial_hidden = mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        mac_out, hidden_store, local_qs = mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach())
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)      
        #-------------------------------------------------------------
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()
                    
        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        #target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
        # for cds----------------------------------------------------
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]).to(self.args.device)
        target_mac_out, _, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach())
        target_mac_out = target_mac_out[:, 1:]
		#------------------------------------------------------------

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(
                3).shape + (self.n_actions,)).to(self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(
                3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(
                target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]
            
		# for cds intrinsic incentive ----------------------------------------------------
        with th.no_grad(): # turn off: autograd engine

            obs = batch["obs"][:, :-1]
            obs_next = batch["obs"][:, 1:]
            mask_clone = mask.detach().clone(
            ).unsqueeze(-2).expand(obs.shape[:-1] + mask.shape[-1:])
            mask_clone = mask_clone.permute(0, 2, 1, 3)
            mask_clone = mask_clone.reshape(-1,
                                            mask_clone.shape[-2], mask_clone.shape[-1])
            mask_clone = mask_clone.reshape(-1, mask_clone.shape[-1])

            obs_intrinsic = obs.clone().permute(0, 2, 1, 3)
            obs_intrinsic = obs_intrinsic.reshape(
                -1, obs_intrinsic.shape[-2], obs_intrinsic.shape[-1])
            eval_h_intrinsic = hidden_store.clone().permute(0, 2, 1, 3)
            eval_h_intrinsic = eval_h_intrinsic.reshape(
                -1, eval_h_intrinsic.shape[-2], eval_h_intrinsic.shape[-1])
            h_cat = th.cat([initial_hidden.reshape(-1, initial_hidden.shape[-1]
                                                   ).unsqueeze(1), eval_h_intrinsic[:, :-2]], dim=1)
            add_id = th.eye(self.args.n_agents).to(obs.device).expand([obs.shape[0], obs.shape[1], self.args.n_agents,
                                                                       self.args.n_agents]).permute(0, 2, 1, 3)

            actions_onehot_clone = actions_onehot.clone().permute(0, 2, 1, 3)

            intrinsic_input_1 = th.cat(
                [h_cat, obs_intrinsic,
                 actions_onehot_clone.reshape(-1, actions_onehot_clone.shape[-2], actions_onehot_clone.shape[-1])],
                dim=-1)

            intrinsic_input_2 = th.cat(
                [intrinsic_input_1, add_id.reshape(-1, add_id.shape[-2], add_id.shape[-1])], dim=-1)

            intrinsic_input_1 = intrinsic_input_1.reshape(
                -1, intrinsic_input_1.shape[-1])
            intrinsic_input_2 = intrinsic_input_2.reshape(
                -1, intrinsic_input_2.shape[-1])

            next_obs_intrinsic = obs_next.clone().permute(0, 2, 1, 3)
            next_obs_intrinsic = next_obs_intrinsic.reshape(
                -1, next_obs_intrinsic.shape[-2], next_obs_intrinsic.shape[-1])
            next_obs_intrinsic = next_obs_intrinsic.reshape(
                -1, next_obs_intrinsic.shape[-1])

            log_p_o = self.target_predict_withoutid.get_log_pi(
                intrinsic_input_1, next_obs_intrinsic)
            log_q_o = self.target_predict_withid.get_log_pi(
                intrinsic_input_2, next_obs_intrinsic, add_id.reshape([-1, add_id.shape[-1]]))

            mean_p = th.softmax(mac_out[:, :-1], dim=-1).mean(dim=2) # mac_out(Q):[32,30,6,14] -->softmax --> pi(a_t|tau_t,id) -->  P(a_t|tau_t) ~= pi(a_t|tau_t,id).mean(dim=2)
            q_pi = th.softmax(self.args.beta1 * mac_out[:, :-1], dim=-1) # 

            # D_KL(SoftMax(Beta1*Q || P(a_t||tau_t)
            pi_diverge = th.cat(
                [(q_pi[:, :, id] * th.log(q_pi[:, :, id] / mean_p)).sum(dim=-
                                                                        1, keepdim=True) for id in range(self.args.n_agents)],
                dim=-1).permute(0, 2, 1).unsqueeze(-1)

            intrinsic_rewards_cds = self.args.beta1 * log_q_o - log_p_o # observation aware diversity
            intrinsic_rewards_cds = intrinsic_rewards_cds.reshape(
                -1, obs_intrinsic.shape[1], intrinsic_rewards_cds.shape[-1])
            intrinsic_rewards_cds = intrinsic_rewards_cds.reshape(
                -1, obs.shape[2], obs_intrinsic.shape[1], intrinsic_rewards_cds.shape[-1])
            intrinsic_rewards_cds = intrinsic_rewards_cds + self.args.beta2 * pi_diverge

            if self.args.anneal:
                if t_env > 1000000:
                    intrinsic_rewards_cds = max(
                        1 - self.args.anneal_rate * (t_env - 1000000) / 1000000, 0) * intrinsic_rewards_cds

        # update predict network
        add_id = add_id.reshape([-1, add_id.shape[-1]])
        for index in BatchSampler(SubsetRandomSampler(range(intrinsic_input_1.shape[0])), 256, False):
            self.eval_predict_withoutid.update(
                intrinsic_input_1[index], next_obs_intrinsic[index], mask_clone[index])
            self.eval_predict_withid.update(
                intrinsic_input_2[index], next_obs_intrinsic[index], add_id[index], mask_clone[index])
		#------------------------------------------------------------
		
        # Mix
        if 'academy' in self.args.env:
            additional_input = obs=batch["obs"][:, :-1]  # for cds_gfootball
        else:
            additional_input = None

        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    mixer(chosen_action_qvals,
                          batch["state"][:, :-1], is_v=True, obs= additional_input)
                ans_adv, _, _ = mixer(
                    chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot, max_q_i=max_action_qvals, is_v=False, obs= additional_input)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals,
                                   batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1],
                                actions=actions_onehot, max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(
                        target_chosen_qvals, batch["state"][:, 1:], is_v=True, obs= additional_input)
                    target_adv, _, _ = self.target_mixer(
                        target_chosen_qvals, batch["state"][:, 1:], actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False, obs= additional_input)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(
                        target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(
                        target_chosen_qvals, batch["state"][:, 1:], actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(
                    target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        if self.args.use_emdqn and self.use_AEM:
            ec_buffer.update_counter += 1
            qec_input = chosen_action_qvals.clone().detach()
            qec_input_new = []
            eta  = th.zeros_like(qec_input).detach().to(self.args.device)

            Vopt = target_max_qvals.clone().detach() # default value # start from s[t+1]               

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
          
        #targets = float(self.args.optimality_incentive)*self.args.gamma*eta + intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        
        if self.args.optimality_incentive:
            targets = self.args.gamma*eta + intrinsic_rewards+rewards + self.args.beta * intrinsic_rewards_cds.mean(dim=1) + self.args.gamma * (1 - terminated) * target_max_qvals        
        else:
            targets = intrinsic_rewards+rewards + self.args.beta * intrinsic_rewards_cds.mean(dim=1) + self.args.gamma * (1 - terminated) * target_max_qvals        

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
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss

        
		# for cds-------------------------------------------
        norm_loss = F.l1_loss(local_qs, target=th.zeros_like(
            local_qs), size_average=True)
        loss += norm_loss / 10 # 10 : fixed scaling factor
        # --------------------------------------------------

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)            
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)            
            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False, ec_buffer=None):

        #intrinsic_rewards = self.vdn_learner.train(batch, t_env, episode_num,save_buffer=False, imac=self.mac, timac=self.target_mac)
        intrinsic_rewards = 0.0
        if self.args.is_prioritized_buffer:
            masked_td_error, mask = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)
        else:
            self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(ec_buffer)
         
            self.last_target_update_episode = episode_num
            
        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res

    def _update_targets(self,ec_buffer):
        if self.args.use_emdqn:
            ec_buffer.update_kdtree()
        self.target_mac.load_state(self.mac)
        self.target_predict_withid.load_state_dict(
            self.eval_predict_withid.state_dict())
        self.target_predict_withoutid.load_state_dict(
            self.eval_predict_withoutid.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        #self.vdn_learner.cuda()
        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)

    def save_models(self, path, ec_buffer): ## save models from here...
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.eval_predict_withid.state_dict(),
                "{}/pid.th".format(path))
        th.save(self.eval_predict_withoutid.state_dict(),
                "{}/poid.th".format(path))
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
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.eval_predict_withid.load_state_dict(
            th.load("{}/pid.th".format(path), map_location=lambda storage, loc: storage))
        self.target_predict_withid.load_state_dict(
            th.load("{}/pid.th".format(path), map_location=lambda storage, loc: storage))
        self.eval_predict_withoutid.load_state_dict(
            th.load("{}/poid.th".format(path), map_location=lambda storage, loc: storage))
        self.target_predict_withoutid.load_state_dict(
            th.load("{}/poid.th".format(path), map_location=lambda storage, loc: storage))
