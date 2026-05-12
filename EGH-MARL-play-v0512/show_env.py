"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappoTest",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "mappo",
            "mappoTest"
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        # default="pettingzoo_mpe",
        default="rendezvous",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "rendezvous"
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

    # if args["env"] == "dexhands":
    #     import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    trajectories, pooling_plan = runner.eval(episodes=512)
    runner.close()
    pos_x_list = []
    pos_y_list = []
    ori__list = []
    group = []


    for p in pooling_plan:
        indices = np.nonzero(p)
        group.append(indices[1])

    for t in trajectories:
        pos = t[0]['pos'][:]
        temp_ori = t[0]['ori']
        temp_pos_x = pos[:,0]
        temp_pos_y = pos[:,1]
        pos_x_list.append(temp_pos_x)
        pos_y_list.append(temp_pos_y)
        ori__list.append(temp_ori)
    make_ani(pos_x_list, pos_y_list, ori__list, group)
    
    

def make_ani(pos_x_list, pos_y_list, ori__list, group):
    matplotlib.use('Agg')
    print(len(pos_x_list))
    # 创建一些随机的初始数据
    num_agents = 10
    show_index = 0
    x = pos_x_list[show_index]
    y = pos_y_list[show_index]
    g = group[show_index]
    # 创建随机的朝向向量（长度为arrow_length）
    arrow_length = 4
    angles = ori__list[show_index]
    # angles = np.random.rand(num_agents) * 2 * np.pi
    dx = arrow_length * np.cos(angles)
    dy = arrow_length * np.sin(angles)
    # 创建一个散点图和箭头图
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=g)
    arrows = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
    # fig.savefig('0.png')

    #  # 设置x和y轴范围
    # padding = 10  # 调整范围的padding大小
    # x_min, x_max = np.min(pos_x_list) - padding, np.max(pos_x_list) + padding
    # y_min, y_max = np.min(pos_y_list) - padding, np.max(pos_y_list) + padding
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)

    # 更新函数，每次调用都会更新图表
    def update(frame):
        show_index = frame % len(pos_x_list)  # 使用取余运算防止越界
        x = pos_x_list[show_index]
        y = pos_y_list[show_index]
        g = group[show_index]
        
        # angles = np.random.rand(num_agents) * 2 * np.pi
        angles = ori__list[show_index]
        show_index += 1
        dx = arrow_length * np.cos(angles)
        dy = arrow_length * np.sin(angles)

        sc.set_offsets(np.c_[x, y])
        sc.set_array(g)
        arrows.set_offsets(np.c_[x, y])
        arrows.set_UVC(dx, dy)
        # fig.savefig('figures/{}.png'.format(show_index))
        # print(g.tolist())
        # print("group1: {}\ngroup2: {}\ngroup3: {}\n".format(np.where(g==0)[0].tolist(), np.where(g==1)[0].tolist(), np.where(g==2)[0].tolist()))
    ani = animation.FuncAnimation(fig, update, frames=range(len(pos_x_list)), interval=50)
    ani.save('animation.gif', writer='pillow')  # 保存为GIF文件
    # plt.show()





if __name__ == "__main__":
    main()
    