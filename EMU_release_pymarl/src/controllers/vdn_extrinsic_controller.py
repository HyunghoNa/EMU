from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class vdn_extrinsic_controller:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)


    def forward(self, ep_batch):
        inputs =ep_batch["state"]
        agent_inputs=inputs.contiguous().view(-1,self.input_shape)
        agent_outs = self.agent(agent_inputs)

        # Softmax the agent outputs if they're policy logits

        return agent_outs.contiguous().view(ep_batch.batch_size, -1, self.args.rnd_predict_dim)

    #def init_hidden(self, batch_size):
     #   self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

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
        self.agent = agent_REGISTRY[self.args.rnd_predict_agent](input_shape, self.args)


    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        return input_shape
