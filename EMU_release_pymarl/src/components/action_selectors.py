import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        if args.env == 'mmdp_game_1' and args.joint_random_policy_eps > 0:
            self.joint_eps = args.joint_random_policy_eps
            if args.is_1_4:
                self.joint_action_seeds = th.Tensor([1., 0., 0., 1.])
            else:
                self.joint_action_seeds = th.Tensor([0., 1., 1., 0.])

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        if self.args.env == 'mmdp_game_1' and self.args.joint_random_policy_eps > 0:
            joint_random_seeds = Categorical(th.unsqueeze(self.joint_action_seeds, 0).repeat(agent_inputs.shape[0], 1)).sample().long()
            joint_random_actions = th.zeros_like(agent_inputs[:, :, 0])
            joint_random_actions[:, 0] = joint_random_seeds[:] // 2
            joint_random_actions[:, 1] = joint_random_seeds[:] % 2
            joint_random_numbers = th.rand_like(agent_inputs[:, 0, 0])
            joint_pick_random = th.unsqueeze((joint_random_numbers < self.joint_eps).long(), 1).repeat(1, 2)
            picked_actions = joint_pick_random * joint_random_actions.long() + (1 - joint_pick_random) * picked_actions.long()

        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
