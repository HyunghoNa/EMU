REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_sd_agent import RNN_SD_Agent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .sco_agent import SCOAgent
from .simple_agent import SimpleAgent
from .rnd_nn_agent import RND_nn_Agent
from .rnn_agent import RNNAgent
from .rnn_fast_agent import RNNFastAgent
from .rnd_history_agent import RNDHistoryAgent
from .rnn_individualQ_agent import RNN_individualQ_Agent
from .rnd_fast_history_agent import RND_Fast_historyAgent



REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_sd"] = RNN_SD_Agent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["rnn_fast"] = RNNFastAgent
REGISTRY["sco"] = SCOAgent
REGISTRY["simple"] = SimpleAgent
REGISTRY["rnd_nn"] = RND_nn_Agent
REGISTRY["rnd_history"] = RNDHistoryAgent
REGISTRY["rnn_individualQ"] = RNN_individualQ_Agent
REGISTRY["rnd_fast_history"] = RND_Fast_historyAgent
