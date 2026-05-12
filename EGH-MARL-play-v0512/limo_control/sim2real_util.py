from nokov.nokovsdk import *
import numpy as np
from math import *
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

# 四元数转欧拉角(x,y,z)
def quaternion_to_euler(qx,qy,qz,qw):
    roll = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = asin(2 * (qw * qy - qx * qz))
    yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))
    return roll, pitch, yaw

def map_to_virtual(real_point, real_corners, world_size):
    virtual_point = np.zeros((1, 2))
    bl, br, tr, tl = real_corners
    bl = np.array(bl)
    br = np.array(br)
    tr = np.array(tr)
    tl = np.array(tl)
    x_rate = float(br[0] - bl[0])
    y_rate = float(tl[1] - bl[1])
    x = real_point[0]
    x = np.clip(x, bl[0], br[0])
    y = real_point[1]
    y = np.clip(y, bl[1], tl[1])
    virtual_point[0, 0] = (x - bl[0]) / x_rate * world_size
    virtual_point[0, 1] = (y - bl[1]) / y_rate * world_size
    return virtual_point

def map_to_real(virtual_point, real_corners, world_size):
    real_point = np.zeros((1, 2))
    bl, br, tr, tl = real_corners
    bl = np.array(bl)
    br = np.array(br)
    tr = np.array(tr)
    tl = np.array(tl)
    x_rate = float(br[0] - bl[0])
    y_rate = float(tl[1] - bl[1])
    x = virtual_point[0]
    y = virtual_point[1]
    real_point[0] = x / world_size * x_rate + bl[0]
    real_point[1] = y / world_size * y_rate + bl[1]
    return real_point

def init_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="egnn_mappo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
            "eghn_mappo",
            "egnn_mappo",
            "eghn_maddpg",
            "gat_mappo",
            "gcn_mappo",
            "graphsage_mappo",
            "mappo_data_aug"
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pursuit",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "smacv2",
            "lag",
            "rendezvous",
            "pursuit",
            "navigation",
            "navigation_v2",
            "cover"
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag, rendezvous.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--my_seed",
        type=int,
        default=2,
        help="set seed",
    )
    args, unparsed_args = parser.parse_known_args()
    def process(arg):
        try:
            return eval(arg)
        except:
            return arg
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    algo_args["seed"]["seed"] = args["my_seed"]
    return args, algo_args, env_args