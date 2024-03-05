from functools import partial
# do not import SC2 in labtop

import socket
if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname():
    from smac.env import MultiAgentEnv, StarCraft2Env
else:
    from .multiagentenv import MultiAgentEnv
import sys
import os
from .grf import Academy_3_vs_1_with_Keeper, Academy_Counterattack_Easy, Academy_Counterattack_Hard

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
} if 'MBP' not in socket.gethostname() and 'DESIGNARE' not in socket.gethostname() else {}
REGISTRY["academy_3_vs_1_with_keeper"]= partial(env_fn, env=Academy_3_vs_1_with_Keeper)
REGISTRY["academy_counterattack_easy"]= partial(env_fn, env=Academy_Counterattack_Easy)
REGISTRY["academy_counterattack_hard"]= partial(env_fn, env=Academy_Counterattack_Hard)


#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
