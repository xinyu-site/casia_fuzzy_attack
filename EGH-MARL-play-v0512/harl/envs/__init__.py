from absl import flags
from harl.envs.smac.smac_logger import SMACLogger
from harl.envs.smacv2.smacv2_logger import SMACv2Logger
from harl.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from harl.envs.gym.gym_logger import GYMLogger
from harl.envs.football.football_logger import FootballLogger
from harl.envs.dexhands.dexhands_logger import DexHandsLogger
from harl.envs.lag.lag_logger import LAGLogger
from harl.envs.ma_envs.rendezvous_logger import RendezvousLogger
from harl.envs.ma_envs.pursuit_logger import PursuitLogger
from harl.envs.ma_envs.navigation_logger import NavigationLogger
from harl.envs.ma_envs.navigation_v2_logger import NavigationV2Logger
from harl.envs.ma_envs.cover_logger import CoverLogger
from harl.envs.mujoco3d.mujoco3d_logger import MuJoCo3dLogger
FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "lag": LAGLogger,
    "rendezvous": RendezvousLogger,
    "pursuit": PursuitLogger,
    "navigation": NavigationLogger,
    "navigation_v2": NavigationV2Logger,
    "cover": CoverLogger,
    "mujoco3d": MuJoCo3dLogger
}
