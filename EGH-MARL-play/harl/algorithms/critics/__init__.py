"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.data_aug_critic import VCritic_data_aug
from harl.algorithms.critics.EghnVCritic import EghnVCritic
from harl.algorithms.critics.EgnnVCritic import EgnnVCritic
from harl.algorithms.critics.Egnnv2VCritic import Egnnv2VCritic
from harl.algorithms.critics.Egnnv3VCritic import Egnnv3VCritic
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from harl.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from harl.algorithms.critics.discrete_q_critic import DiscreteQCritic
from harl.algorithms.critics.eghn_q_critic import EghnQCritic
from harl.algorithms.critics.egnn_q_critic import EgnnQCritic
from harl.algorithms.critics.GatVCritic import GATVCritic
from harl.algorithms.critics.GCNVCritic import GCNVCritic
from harl.algorithms.critics.GraphSAGEVCritic import GraphSAGEVCritic
from harl.algorithms.critics.HieVCritic import HieVCritic
from harl.algorithms.critics.HamaVCritic import HamaVCritic
from harl.algorithms.critics.HmfVCritic import HmfVCritic
from harl.algorithms.critics.HepnVCritic import HepnVCritic
from harl.algorithms.critics.Eghnv2VCritic import Eghnv2VCritic
from harl.algorithms.critics.hmf_critic import HMFDummyCritic
from harl.algorithms.critics.Egnnv2VCriticEval import Egnnv2VCriticEval

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "gat_mappo": GATVCritic,
    "gcn_mappo": GCNVCritic,
    "haddpg": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hasac": SoftTwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "matd3": TwinContinuousQCritic,
    "eghn_mappo": EghnVCritic,
    "eghnv2_mappo": Eghnv2VCritic,
    "eghn_critic_mappo": Eghnv2VCritic,
    "eghn_critic_happo": Eghnv2VCritic,
    "eghn_actor_mappo": VCritic,
    "eghn_maddpg": EghnQCritic,
    "egnn_mappo": EgnnVCritic,
    "egnnv2_mappo": Egnnv2VCritic,
    "egnn_mix_mappo": Egnnv2VCritic,
    "egnnv3_mappo": Egnnv3VCritic,
    "egnn_actor_mappo": Egnnv3VCritic,
    'egnn_critic_mappo': Egnnv3VCritic,
    'egnn_critic_happo': Egnnv3VCritic,
    "graphsage_mappo": GraphSAGEVCritic,
    "mappo_data_aug": VCritic_data_aug,
    "egnn_maddpg": EgnnQCritic,
    "hie_mappo": HieVCritic,
    "hie_critic_mappo": HieVCritic,
    "hama_mappo": HamaVCritic,
    "hmf_mappo": HmfVCritic,
    "hepn_mappo": HepnVCritic,
    "hmf": HMFDummyCritic,
}

EVAL_CRITIC_REGISTRY = {
    "egnnv2_mappo": Egnnv2VCriticEval,
    "egnn_mix_mappo": Egnnv2VCriticEval,
}
