"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.matd3 import MATD3
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.eghn_mappo import EGHNMAPPO
from harl.algorithms.actors.eghn_v2_mappo import EGHNv2MAPPO
from harl.algorithms.actors.eghn_maddpg import EGHNDDPG

from harl.algorithms.actors.egnn_mappo import EGNNMAPPO
from harl.algorithms.actors.egnn_v2_mappo import EGNNv2MAPPO
from harl.algorithms.actors.egnn_mix_mappo import EGNNMIXMAPPO
from harl.algorithms.actors.egnn_v3_mappo import EGNNv3MAPPO
from harl.algorithms.actors.egnn_maddpg import EGNNDDPG

from harl.algorithms.actors.gat_mappo import GATMAPPO
from harl.algorithms.actors.gcn_mappo import GCNMAPPO
from harl.algorithms.actors.graphsage_mappo import GrapgSAGEMAPPO
from harl.algorithms.actors.data_aug_mappo import MAPPO_data_aug
from harl.algorithms.actors.hie_mappo import HieMAPPO
from harl.algorithms.actors.hama_mappo import HamaMAPPO
from harl.algorithms.actors.hmf_mappo import HmfMAPPO
from harl.algorithms.actors.hepn_mappo import HepnMAPPO

from harl.algorithms.actors.hmf import HMF

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
    "eghn_mappo": EGHNMAPPO,
    "eghnv2_mappo": EGHNv2MAPPO,
    "eghn_critic_mappo": MAPPO,
    "eghn_critic_happo": HAPPO,
    "eghn_actor_mappo": EGHNv2MAPPO,
    "eghn_maddpg": EGHNDDPG,
    "egnn_mappo": EGNNMAPPO,
    "egnnv2_mappo": EGNNv2MAPPO,
    "egnn_mix_mappo": EGNNMIXMAPPO,
    "egnnv3_mappo": EGNNv3MAPPO,
    "egnn_actor_mappo": EGNNv3MAPPO,
    "egnn_critic_mappo": MAPPO,
    "egnn_critic_happo": HAPPO,
    "gat_mappo": GATMAPPO,
    "gcn_mappo": GCNMAPPO,
    "graphsage_mappo": GrapgSAGEMAPPO,
    "mappo_data_aug": MAPPO_data_aug,
    "egnn_maddpg": EGNNDDPG,
    "hie_mappo": HieMAPPO,
    "hie_critic_mappo": MAPPO,
    "hama_mappo": HamaMAPPO,
    "hmf_mappo": HmfMAPPO,
    "hepn_mappo": HepnMAPPO,
    "hmf": HMF,
}
