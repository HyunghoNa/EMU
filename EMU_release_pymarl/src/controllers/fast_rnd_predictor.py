from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class RNDfastpredictor:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None



    def forward(self, ep_batch, t, test_mode=False, batch_inf=False):
        agent_inputs = self._build_inputs(ep_batch, t, batch_inf)
        epi_len = t if batch_inf else 1
        avail_actions = ep_batch["avail_actions"][:, :t] if batch_inf else ep_batch["avail_actions"][:, t:t+1]
        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            agent_outs, self.hidden_states, individual_Q = self.agent(agent_inputs, self.hidden_states)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.transpose(1, 2).reshape(ep_batch.batch_size * self.n_agents, epi_len, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=-1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        if hasattr(self.args, 'use_individual_Q') and self.args.use_individual_Q:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), individual_Q.view(ep_batch.batch_size, self.n_agents, -1)
        else:
            if batch_inf:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, epi_len, -1).transpose(1, 2)
            else:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def to(self, *args, **kwargs):
        self.agent.to(*args, **kwargs)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY["rnd_fast_history"](input_shape, self.args)

    def _build_inputs(self, batch, t, batch_inf):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        if batch_inf:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, :t])  # bTav
            if self.args.obs_last_action:
                last_actions = th.zeros_like(batch["actions_onehot"][:, :t])
                last_actions[:, 1:] = batch["actions_onehot"][:, :t-1]
                inputs.append(last_actions)
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).view(1, 1, self.n_agents, self.n_agents).expand(bs, t, -1, -1))

            inputs = th.cat([x.transpose(1, 2).reshape(bs*self.n_agents, t, -1) for x in inputs], dim=2)
            return inputs
        else:
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, t])  # b1av
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

            inputs = th.cat([x.reshape(bs*self.n_agents, 1, -1) for x in inputs], dim=2)
            return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
