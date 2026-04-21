"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import matplotlib.pyplot as plt
import os
import re

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="egnnv2_mappo",
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
            "eghn_mappo",     # 原始的EGHN
            "eghnv2_mappo",   # 可以处理多个等变特征
            "eghn_critic_mappo",  # critic使用EGHNv2
            "eghn_critic_happo",  # critic使用EGHNv2, actor使用HAPPO
            "eghn_actor_mappo",   # actor使用EGHNv2
            "egnn_actor_mappo",   # actor使用EGNNv3
            "egnn_critic_mappo",  # critic使用EGNNv3
            "egnn_critic_happo",
            "egnn_mappo",     # 原始论文的EGNN
            "egnnv2_mappo",   # EGHN论文里的EGNN
            "egnn_mix_mappo",   # 混合EGNN
            "egnnv3_mappo",   # 可以处理多个等变特征
            "gat_mappo",
            "gcn_mappo",
            "graphsage_mappo",
            "mappo_data_aug",
            "hie_mappo",
            'hie_critic_mappo',
            "hepn_mappo",      # 结构熵分层
            "hmf_mappo", # "hmf",
            "hama_mappo"
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="navigation",
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
            "cover",
            "mujoco3d"
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
        "--results_dir",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--attack_method",
        type=str,
        default="none",
        choices=[
            "none",
            "obs_grd_single",
            "obs_noise_single",
            "obs_grd_all",
            "obs_noise_all",
            "act_rotation_single",
            "act_noise_single",
            "act_rotation_all",
            "act_noise_all",
            #"obs_rotation_all"
            ],
        help="Attack method. Choose from: obs_grd_single, obs_noise_single, obs_grd_all, obs_noise_all, act_rotation_single, act_noise_single, act_rotation_all, act_noise_all.",
    )

    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise level for attack method that requires noise. Default 0.0.",
    )

    parser.add_argument(
        "--noise_num",
        type=float,
        default=0.0,
        help="Noise number for attack method that requires noise. Default 0.0.",
    )

    parser.add_argument(
        "--env_num1",
        type=float,
        default=0.0,
        help="Environment parameter 1, default 0.0."
    )

    parser.add_argument(
        "--env_num2",
        type=float,
        default=0.0,
        help="Environment parameter 2, default 0.0."
    )

    parser.add_argument(
        "--episode",
        type = int,
        default = 100,
        help = "Number of episodes to evaluate."
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
    env_args["env_num1"] = args["env_num1"]
    env_args["env_num2"] = args["env_num2"]
    attack_method = args["attack_method"]
    noise_level = args["noise_level"] if "noise_level" in args else 0.0
    noise_num = args["noise_num"] if "noise_num" in args else 0.0

    # if args["env"] == "dexhands":
    #     import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start evaluation
    from harl.runners import ATTACK_RUNNER_REGISTRY
    # algo_args["seed"]["seed"] = args["my_seed"]
    model_path = f'{args["results_dir"]}/{args["env"]}/{env_args["obs_mode"].replace("_", "")}-{env_args["dynamics"].replace("_", "")}/{args["algo"]}/{args["exp_name"]}'
    #print(f"Model path: {model_path}")
    # runner = EVAL_RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args,model_path)
    # if algo_args["train"]["train_flag"]:
    #     runner.run()
    # 获取当前工作目录的绝对路径
    current_dir = os.getcwd()

    # 合成完整路径（在当前文件夹下）
    full_path = os.path.join(current_dir, model_path)

    # 检查路径是否存在
    if os.path.exists(full_path):
        sub_folders = [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
        print(f"path exist: {full_path}")
        #print(f"子文件夹列表: {sub_folders}")
    else:
        print(f"path not exist: {full_path}")
    pattern = r'seed-(\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'
    strict_matched_results = []
    matched_results = []
    for folder_name in sub_folders:
        match = re.search(pattern, folder_name)
        if match:
            seed = match.group(1)
            timestamp = match.group(2)
            print(f"Found seed: {seed}, timestamp: {timestamp} in folder name: {folder_name}")
            #print('seed {}, algo_args["seed"] {}'.format(seed, algo_args["seed"]))
            if int(seed) == int(algo_args["seed"]['seed']):
                strict_matched_results.append((folder_name, seed, timestamp))
                matched_results.append((folder_name, seed, timestamp))
            else:
                matched_results.append((folder_name, seed, timestamp))
        else:
            print(f"No seed and timestamp found in folder name: {folder_name}")
    if strict_matched_results:
        strict_matched_results.sort(key=lambda x: x[2], reverse=False)
        selected_folder = strict_matched_results[0][0]
        print(f"Selected folder based on strict match: {selected_folder}")
    elif matched_results:
        matched_results.sort(key=lambda x: x[2], reverse=False)
        selected_folder = matched_results[0][0]
        print(f"Selected folder based on match: {selected_folder}")
    else:
        print("No matching folders found based on seed.")
        return
    model_path = os.path.join(full_path, selected_folder, "models/")
    print(f"Final model path: {model_path}")
    runner = ATTACK_RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args, model_path)
    
    # episodes = (
    #     int(algo_args["train"]["num_env_steps"])
    #     // algo_args["train"]["episode_length"]
    #     // algo_args["train"]["n_rollout_threads"]
    # )

    episodes = args["episode"]
    
    if algo_args["train"]["train_flag"]:
        aver_reward = runner.eval(episodes, attack_method=attack_method, noise_level=noise_level, noise_num=noise_num)
    runner.close()


if __name__ == "__main__":
    print("Playing...")
    main()