REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_offpolicy_runner import EpisodeRunner as OffPolicyRunner
REGISTRY["offpolicy"] = OffPolicyRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
