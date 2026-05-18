from .envs.point_envs.navigation import NavigationEnv_ori
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from torch.nn.functional import cosine_similarity
import networkx as nx
from harl.utils.trans_graph import trans_discon, extract_layer_data
import torch

class NavigationEnv:
    def __init__(self, args):
        self.args = args
        self.env = NavigationEnv_ori(local_mode=self.args["local_mode"],
                                    obs_radius=self.args["obs_radius"],
                                    windows_size=self.args["windows_size"],
                                    use_history=self.args["use_history"],
                                    nr_pursuers=self.args["nr_agents"],
                                    int_points_num=self.args["int_points_num"],
                                    obs_mode=self.args["obs_mode"],
                                    comm_radius=self.args["comm_radius"],
                                    world_size=self.args["world_size"],
                                    distance_bins=self.args["distance_bins"],
                                    bearing_bins=self.args["bearing_bins"],
                                    torus=self.args["torus"],
                                    dynamics=self.args["dynamics"],
                                    env_num1=self.args["env_num1"],
                                    env_num2=self.args["env_num2"])
        self.state_type = self.args["state_type"]
        self.n_agents = self.env.nr_agents
        # self.env.reset()
        self.agents = self.env.world.agents
        if self.state_type == "EP":
            self.share_observation_space = self.unwrap(self.env.share_observation_space)
        else:
            self.share_observation_space = self.unwrap(self.env.observation_space)
        self.observation_space = self.unwrap(self.env.observation_space)
        self.action_space = self.unwrap(self.env.action_space)

    def step(self, actions):
        obs, s_obs, rewards, done, info, avaliable_actions = self.env.step(actions) 
        if self.state_type == "EP":
            rewards = np.ones((self.n_agents, 1)) * rewards.mean()
            if self.args["structural_entropy"]:
                obs_ = np.array(obs)
                G = nx.Graph()
                G.add_nodes_from(range(self.n_agents))
                if self.args["local_mode"]:
                    adj = obs_[:, -self.n_agents:]
                    edges = self.get_edges(adj)
                    edge_weights = cosine_similarity(torch.tensor(obs_[edges[0], self.args["equ_nf"] : -self.n_agents]),
                                                     torch.tensor(obs_[edges[1], self.args["equ_nf"] : -self.n_agents]))
                else:
                    edges = self.get_edges()
                    edge_weights = cosine_similarity(torch.tensor(obs_[edges[0], self.args["equ_nf"]:]),
                                                     torch.tensor(obs_[edges[1], self.args["equ_nf"]:]))

                edge_weights = torch.exp(edge_weights)
                # edge_weights[edge_weights < 0] = 0
                edge_weights = edge_weights.tolist()
                G.add_weighted_edges_from([[r, c, w] for r, c, w in zip(edges[0], edges[1], edge_weights)])
                trees = trans_discon((G, 2))
                layer_data = extract_layer_data(trees, G, 2)
                return (obs, s_obs, np.array(layer_data), rewards, self.repeat(done), self.repeat(info), avaliable_actions)
            return (
                obs,
                s_obs,
                rewards,
                self.repeat(done),
                self.repeat(info),
                avaliable_actions
            )
        else:
            return (
                obs,
                obs,
                rewards,
                self.repeat(done),
                self.repeat(info),
                avaliable_actions
            )
    
    def unwrap(self, d):
        l = []
        for i in range(self.n_agents):
            l.append(d[i])
        return l 
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def reset(self):
        obs, s_obs, available_actions = self.env.reset()
        if self.state_type == "EP":
            if self.args["structural_entropy"]:
                obs_ = np.array(obs)
                G = nx.Graph()
                G.add_nodes_from(range(self.n_agents))
                if self.args["local_mode"]:
                    adj = obs_[:, -self.n_agents:]
                    edges = self.get_edges(adj)
                    edge_weights = cosine_similarity(torch.tensor(obs_[edges[0], self.args["equ_nf"] : -self.n_agents]),
                                                     torch.tensor(obs_[edges[1], self.args["equ_nf"] : -self.n_agents]))
                else:
                    edges = self.get_edges()
                    edge_weights = cosine_similarity(torch.tensor(obs_[edges[0], self.args["equ_nf"]:]),
                                                     torch.tensor(obs_[edges[1], self.args["equ_nf"]:]))
                edge_weights = torch.exp(edge_weights)
                # edge_weights = torch.clamp(edge_weights, min=0.0)
                edge_weights = edge_weights.tolist()
                G.add_weighted_edges_from([[r, c, w] for r, c, w in zip(edges[0], edges[1], edge_weights)])
                trees = trans_discon((G, 2))
                layer_data = extract_layer_data(trees, G, 2)
                return obs, s_obs, np.array(layer_data), available_actions
            return obs, s_obs, available_actions
        else:
            return obs, obs, available_actions
        
    def ue_set(self, states):
        obs, s_obs, available_actions = self.env.ue_set(states)
        if self.state_type == "EP":
            return obs, s_obs, available_actions
        else:
            return obs, obs, available_actions
    
    def seed(self, seed):
        self.env.seed(seed=seed)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_edges(self, adjacency_matrix=None):
        rows, cols = [], []
        if adjacency_matrix is None:
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
        else:
            n_agents = adjacency_matrix.shape[0]
            for i in range(n_agents):
                for j in range(n_agents):
                    if adjacency_matrix[i, j] == 1 and i != j:
                        rows.append(i)
                        cols.append(j)
        edges = [rows, cols]
        return edges

    def make_ani(self, trajectories):
        from matplotlib.colors import ListedColormap

        tra = trajectories[0]
        pooling_plan = trajectories[3]
        pos_x_list = []
        pos_y_list = []
        ori__list = []
        eva_pos_x_list = []
        eva_pos_y_list = []
        base_colors = [
            '#FF8E98',
            '#3BBCDF',
            '#BB72AA',
            # '#FEB4B9', 
            # '#90D5E8',
            # '#D19FC5',
        ]

        for t in tra:
            temp = t['pursuer_states'][:]
            temp_ori = temp[:, 2]
            temp_pos_x = temp[:,0]
            temp_pos_y = temp[:,1]
            pos_x_list.append(temp_pos_x)
            pos_y_list.append(temp_pos_y)
            ori__list.append(temp_ori)

            temp2 = t['evader_states'][:]
            temp_pos_x2 = temp2[:, 0]
            temp_pos_y2 = temp2[:, 1]
            eva_pos_x_list.append(temp_pos_x2)
            eva_pos_y_list.append(temp_pos_y2)
        matplotlib.use('Agg')
        print(len(pos_x_list))

        # 创建一个散点图和箭头图
        fig, ax = plt.subplots()
        ax.set_aspect('equal')  # 设置横纵比确保圆形看起来是圆的
        ax.set_xlim(0, self.env.world_size)
        ax.set_ylim(0, self.env.world_size)
        # 不显示坐标轴刻度/标签，但保留黑色边框，便于论文排版
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

        sc = None
        arrows = None
        sc1 = None
        trail_scatters = []

        # 更新函数，每次调用都会更新图表
        def update(frame):
            nonlocal sc, arrows, sc1, trail_scatters
            show_index = frame % len(pos_x_list)  # 使用取余运算防止越界
            x = pos_x_list[show_index]
            y = pos_y_list[show_index]
            if len(pooling_plan) > 0:
                # current_plan = pooling_plan[show_index]
                current_plan = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
            x1 = eva_pos_x_list[show_index]
            y1 = eva_pos_y_list[show_index]

            arrow_length = 2
            angles = ori__list[show_index]
            
            dx = arrow_length * np.cos(angles)
            dy = arrow_length * np.sin(angles)

            # 清除尾迹
            for sc_trail in trail_scatters:
                sc_trail.remove()
            trail_scatters = []

            # 绘制尾迹
            window = 15
            alpha_min, alpha_max = 0.01, 1
            size_min, size_max = 1, 30
            for i in range(window, -1, -1):
                idx = show_index - i
                if idx < 0:
                    continue
                px = pos_x_list[idx]
                py = pos_y_list[idx]
                alpha = alpha_max - (alpha_max - alpha_min) * (i / window)
                size = size_max - (size_max - size_min) * (i / window)
                if len(pooling_plan) > 0:
                    num_classes = len(np.unique(current_plan))
                    cmap = ListedColormap(base_colors[:num_classes])
                    sc_trail = ax.scatter(px, py, c=current_plan, cmap=cmap, alpha=alpha, s=size)
                else:
                    sc_trail = ax.scatter(px, py, alpha=alpha, s=size)
                trail_scatters.append(sc_trail)
            # 绘制散点图，根据分组数量动态更新色系
            # if sc is not None:
            #     sc.remove()
            # if len(pooling_plan) > 0:
            #     num_classes = len(np.unique(current_plan))
            #     cmap = ListedColormap(base_colors[:num_classes])
            #     sc = ax.scatter(x, y, c=current_plan, cmap=cmap)
            # else:
            #     sc = ax.scatter(x, y)
            # 绘制箭头
            if arrows is not None:
                arrows.remove()
            arrows = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
            # else:
            #     arrows.set_offsets(np.c_[x, y])
            #     arrows.set_UVC(dx, dy)
            # 绘制目标点
            if sc1 is None:
                sc1 = ax.scatter(x1, y1, c='black',marker = 'o', s=200, alpha=0.5)  
            else:
                sc1.set_offsets(np.c_[x1, y1])
            print(show_index)
            plt.savefig(f'./hie_fig/collection/{show_index:03d}.pdf', bbox_inches='tight', pad_inches=0.02)
            # show_index += 1
        ani = animation.FuncAnimation(fig, update, frames=range(len(pos_x_list)), interval=100)
        ani.save('./navigation_animation.gif', writer='pillow')  # 保存为GIF文件
        # plt.show()
