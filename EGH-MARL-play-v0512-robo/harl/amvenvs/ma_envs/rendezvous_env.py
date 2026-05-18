from .envs.point_envs.rendezvous import RendezvousEnv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import networkx as nx
from harl.utils.trans_graph import trans_discon, extract_layer_data

class RENDEEnv:
    def __init__(self, args):
        args = args
        self.env = RendezvousEnv(local_mode=args["local_mode"],
                                 windows_size=args["windows_size"],
                                 use_history=args["use_history"],
                                 nr_agents=args["nr_agents"],
                                 obs_mode=args["obs_mode"],
                                 comm_radius=args["comm_radius"],
                                 obs_radius=args["obs_radius"],
                                 world_size=args["world_size"],
                                 distance_bins=args["distance_bins"],
                                 bearing_bins=args["bearing_bins"],
                                 torus=args["torus"],
                                 dynamics=args["dynamics"],
                                 env_num1=args["env_num1"],
                                 env_num2=args["env_num2"])
        self.args = args
        self.state_type = args["state_type"]
        self.n_agents = self.env.nr_agents
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

                # edge_weights = torch.exp(edge_weights)
                edge_weights = torch.clamp(edge_weights, min=0.0)
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
    
    # def wrap(self, d):
    #     l = []
    #     for i in range(self.n_agents):
    #         l.append(d[i])
    #     return l
    
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
        # pass
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
        action_list = []
        base_colors = [
            '#FF8E98',
            '#3BBCDF',
            '#BB72AA',
            # '#FEB4B9', 
            # '#90D5E8',
            # '#D19FC5',
        ]

        for t in tra:
            pos = t['pos'][:]
            temp_action = t['ori'] #np.reshape(t['actions'], (t['actions'].shape[1], t['actions'].shape[2]))
            temp_ori = t['ori']
            temp_pos_x = pos[:,0]
            temp_pos_y = pos[:,1]
            pos_x_list.append(temp_pos_x)
            pos_y_list.append(temp_pos_y)
            ori__list.append(temp_ori)
            action_list.append(temp_action)
        matplotlib.use('Agg')
        print(len(pos_x_list))
        # 创建一些随机的初始数据
        
        # 创建一个散点图和箭头图
        fig, ax = plt.subplots()
        ax.set_aspect('equal')  # 设置横纵比确保圆形看起来是圆的
        #  # 设置x和y轴范围
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
        trail_scatters = []

        # 更新函数，每次调用都会更新图表
        def update(frame):
            nonlocal sc, arrows, trail_scatters
            show_index = frame % len(pos_x_list)  # 使用取余运算防止越界
            x = pos_x_list[show_index]
            y = pos_y_list[show_index]
            if len(pooling_plan) > 0:
                current_plan = pooling_plan[show_index]
            
            arrow_length = 5
            angles = ori__list[show_index]
            
            dx = arrow_length * np.cos(angles)
            dy = arrow_length * np.sin(angles)

            # 清除尾迹
            for sc_trail in trail_scatters:
                sc_trail.remove()
            trail_scatters = []
            # 绘制尾迹
            window = 40
            alpha_min, alpha_max = 0.01, 1
            size_min, size_max = 1, 30
            for i in range(window, -1, -2):
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
            
            # 绘制散点图
            # if sc is not None:
            #     sc.remove()
            # if len(pooling_plan) > 0:
            #     num_classes = len(np.unique(current_plan))
            #     cmap = ListedColormap(base_colors[:num_classes])
            #     sc = ax.scatter(x, y, c=current_plan, cmap=cmap)
            # else:
            #     sc = ax.scatter(x, y)
            if arrows is not None:
                arrows.remove()
            arrows = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
            # else:
            #     arrows.set_offsets(np.c_[x, y])
            #     arrows.set_UVC(dx, dy)
            print(show_index)
            plt.savefig(f'./hie_fig/rendezvous/{show_index:03d}.pdf', bbox_inches='tight', pad_inches=0.02)
            # show_index += 1
        ani = animation.FuncAnimation(fig, update, frames=range(len(pos_x_list)), interval=100)
        ani.save('./rendezvous_animation.gif', writer='pillow')  # 保存为GIF文件

