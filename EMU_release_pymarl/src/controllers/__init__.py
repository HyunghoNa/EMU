REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_cds import BasicMAC_cds
from .central_basic_controller import CentralBasicMAC # wqmix
from .fast_controller import FastMAC
from .mmdp_controller import MMDPMAC
from .qsco_controller import qsco_MAC
from .rnd_state_predictor import RND_state_predictor
from .rnd_predictor import RNDpredictor
from .fast_rnd_predictor import RNDfastpredictor
from .state_embedder import StateEmbedder
from .predictV import PredictVCritic
from .vae_embedder import VAE

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_cds"] = BasicMAC_cds
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["fast_mac"] = FastMAC
REGISTRY["mmdp_mac"] = MMDPMAC
REGISTRY["qsco_mac"] = qsco_MAC
REGISTRY["nn_predict"] = RND_state_predictor

REGISTRY["predict"] = RNDpredictor
REGISTRY["fast_predict"] = RNDfastpredictor
REGISTRY["State_Embedder"] = StateEmbedder
REGISTRY["V_predictor"] = PredictVCritic
REGISTRY["VAE"] = VAE