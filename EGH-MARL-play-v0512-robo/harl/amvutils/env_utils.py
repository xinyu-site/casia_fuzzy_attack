import os
import random
import numpy as np
import torch
from harl.amvenvs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecDualEnv, ShareDummyVecDualEnv


def check(value):
    # 异构情况，判断d的dtype是否为object，如果是，则需要转换格式
    if value.dtype == object:
        value = np.array(value[0])
        value = np.expand_dims(value, axis=0)
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    return act_shape


def get_onehot_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = act_space.n
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name == "dexhands":
        from harl.amvenvs.dexhands.dexhands_env import DexHandsEnv

        return DexHandsEnv({"n_threads": n_threads, **env_args})

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.amvenvs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smac_dual":
                from harl.amvenvs.smac.StarCraft2Dual_Env import StarCraft2DualEnv

                env = StarCraft2DualEnv(env_args)
            elif env_name == "smacv2":
                from harl.amvenvs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.amvenvs.mamujoco.mamujoco_env import (
                    MAMujocoEnv,
                )

                env = MAMujocoEnv(env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.amvenvs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                assert env_args["scenario"] in [
                    "simple_spread_v3",
                    "simple_reference_v3",
                    "simple_speaker_listener_v4",
                ], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.amvenvs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.amvenvs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "toy":
                from harl.amvenvs.toy_example.toy_example import ToyExample

                env = ToyExample(env_args)
            elif env_name == "metadrive":
                from harl.amvenvs.metadrive.metadrive_env import MetaDriveEnv

                env = MetaDriveEnv(env_args)
            elif env_name == "quads":
                from harl.amvenvs.quads.quadrotor_multi_env import QuadrotorMultiEnv

                env = QuadrotorMultiEnv(env_args)
            elif env_name == "network":
                from harl.amvenvs.network.network_env import NetworkEnv

                env = NetworkEnv(env_args, rank)
            elif env_name == "voltage":
                from harl.amvenvs.voltage.voltage_env import ValtageEnv

                env = ValtageEnv(env_args)
            elif env_name == "lag":
                from harl.amvenvs.lag.lag_env import LAGEnv
                env = LAGEnv(env_args)
            elif env_name == "rendezvous":
                from harl.amvenvs.ma_envs.rendezvous_env import RENDEEnv
                env = RENDEEnv(env_args)
            elif env_name == "pursuit":
                from harl.amvenvs.ma_envs.pursuit_env import PursuitEnv
                env = PursuitEnv(env_args)
            elif env_name == "navigation":
                from harl.amvenvs.ma_envs.navigation_env import NavigationEnv
                env = NavigationEnv(env_args)
            elif env_name == "navigation_v2":
                from harl.amvenvs.ma_envs.navigation_env_v2 import NavigationV2Env
                env = NavigationV2Env(env_args)
            elif env_name == "cover":
                from harl.amvenvs.ma_envs.cover_env import CoverEnv
                env = CoverEnv(env_args)
            
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if env_name == "smac_dual":
        if n_threads == 1:
            return ShareDummyVecDualEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecDualEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        if n_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":
        from harl.amvenvs.dexhands.dexhands_env import DexHandsEnv

        return DexHandsEnv({"n_threads": n_threads, **env_args})

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.amvenvs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smac_dual":
                from harl.amvenvs.smac.StarCraft2Dual_Env import StarCraft2DualEnv

                env = StarCraft2DualEnv(env_args)
            elif env_name == "smacv2":
                from harl.amvenvs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.amvenvs.mamujoco.mamujoco_env import (
                    MAMujocoEnv,
                )

                env = MAMujocoEnv(env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.amvenvs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.amvenvs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.amvenvs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "toy":
                from harl.amvenvs.toy_example.toy_example import ToyExample

                env = ToyExample(env_args)
            elif env_name == "metadrive":
                from harl.amvenvs.metadrive.metadrive_env import MetaDriveEnv

                env = MetaDriveEnv(env_args)
            elif env_name == "quads":
                from harl.amvenvs.quads.quadrotor_multi_env import QuadrotorMultiEnv

                env = QuadrotorMultiEnv(env_args)
            elif env_name == "network":
                from harl.amvenvs.network.network_env import NetworkEnv

                env = NetworkEnv(env_args, 100 + rank)  # 测试环境的端口号从+100开始，避开训练环境的端口号
            elif env_name == "voltage":
                from harl.amvenvs.voltage.voltage_env import ValtageEnv

                env = ValtageEnv(env_args)
            elif env_name == "rendezvous":
                from harl.amvenvs.ma_envs.rendezvous_env import RENDEEnv
                env = RENDEEnv(env_args)
            elif env_name == "pursuit":
                from harl.amvenvs.ma_envs.pursuit_env import PursuitEnv
                env = PursuitEnv(env_args)
            elif env_name == "lag":
                from harl.amvenvs.lag.lag_env import LAGEnv
                env = LAGEnv(env_args)                
            elif env_name == "navigation":
                from harl.amvenvs.ma_envs.navigation_env import NavigationEnv
                env = NavigationEnv(env_args) 
            elif env_name == "navigation_v2":
                from harl.amvenvs.ma_envs.navigation_env_v2 import NavigationV2Env
                env = NavigationV2Env(env_args)       
            elif env_name == "cover":
                from harl.amvenvs.ma_envs.cover_env import CoverEnv
                env = CoverEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if env_name == "smac_dual":
        if n_threads == 1:
            return ShareDummyVecDualEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecDualEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        if n_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    if env_name == "smac":
        from harl.amvenvs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(args=env_args)
        manual_render = False  # smac does not support manually calling the render() function
        # instead, it use save_replay()
        manual_delay = False
    elif env_name == "smac_dual":
        from harl.amvenvs.smac.StarCraft2Dual_Env import StarCraft2DualEnv

        env = StarCraft2DualEnv(env_args)
        manual_render = False  # smac does not support manually calling the render() function
        # instead, it use save_replay()
        manual_delay = False
    elif env_name == "smacv2":
        from harl.amvenvs.smacv2.smacv2_env import SMACv2Env

        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
    elif env_name == "mamujoco":
        from harl.amvenvs.mamujoco.mamujoco_env import MAMujocoEnv

        env = MAMujocoEnv(env_args)
    elif env_name == "pettingzoo_mpe":
        from harl.amvenvs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
    elif env_name == "gym":
        from harl.amvenvs.gym.gym_env import GYMEnv

        env = GYMEnv(env_args)
    elif env_name == "football":
        from harl.amvenvs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
        manual_render = False  # football renders automatically
    elif env_name == "toy":
        from harl.amvenvs.toy_example.toy_example import ToyExample

        env = ToyExample(env_args)
        manual_render = False
        manual_delay = False
    elif env_name == "dexhands":
        from harl.amvenvs.dexhands.dexhands_env import DexHandsEnv

        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False  # dexhands renders automatically
        manual_expand_dims = False  # dexhands uses parallel envs, thus dimension is already expanded
        manual_delay = False
        env_num = 64
    elif env_name == "network":
        from harl.amvenvs.network.network_env import NetworkEnv

        env = NetworkEnv(env_args)
    elif env_name == "metadrive":
        from harl.amvenvs.metadrive.metadrive_env import MetaDriveEnv

        env = MetaDriveEnv(env_args)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    env.seed(seed * 60000)
    return env, manual_render, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
