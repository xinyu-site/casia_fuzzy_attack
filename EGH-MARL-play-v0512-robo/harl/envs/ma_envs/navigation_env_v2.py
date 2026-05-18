from .envs.point_envs.navigation_v2 import NavigationEnvV2_ori
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib.patches import Circle
import random
import csv

class NavigationV2Env:
    def __init__(self, args):
        self.args = args
        self.env = NavigationEnvV2_ori(local_mode=self.args["local_mode"],
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
                                    dynamics=self.args["dynamics"])
        self.state_type = self.args["state_type"]
        self.n_agents = self.env.nr_agents
        # 障碍物信息
        self.obstacle_radius = self.env.obstacle_radius
        self.int_points_num = self.env.int_points_num
        self.static_obstacle_count = self.env.static_obstacle_count
        self.dynamic_obstacle_count = self.env.dynamic_obstacle_count
        self.env.reset()
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
            return obs, s_obs, available_actions
        else:
            return obs, obs, available_actions
    
    def seed(self, seed):
        self.env.seed(seed=seed)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        
    def make_ani(self, trajectories):
        save_data = False # 保存数据到CSV文件
        trajectories = trajectories[0]
        pos_x_list = []
        pos_y_list = []
        ori__list = []
        eva_pos_x_list = []
        eva_pos_y_list = []

        for t in trajectories:
            # print("vel:", np.mean(t["velocity"]))
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
        # 创建一些随机的初始数据
        show_index = 0
        x = pos_x_list[show_index]
        y = pos_y_list[show_index]
        # 创建随机的朝向向量（长度为arrow_length）
        arrow_length = 4
        angles = ori__list[show_index]
        dx = arrow_length * np.cos(angles)
        dy = arrow_length * np.sin(angles)
        # 创建一个散点图和箭头图
        fig, ax = plt.subplots()
        ax.set_aspect('equal')  # 设置横纵比确保圆形看起来是圆的

        sc = ax.scatter(x, y)
        arrows = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
        x1 = eva_pos_x_list[show_index]
        y1 = eva_pos_y_list[show_index]
        int_point_x = x1[:self.int_points_num]
        int_point_y = y1[:self.int_points_num]
        static_obstacles_x = x1[self.int_points_num:self.int_points_num + self.static_obstacle_count]
        static_obstacles_y = y1[self.int_points_num:self.int_points_num + self.static_obstacle_count]
        dynamic_obstacles_x = x1[-self.dynamic_obstacle_count:]
        dynamic_obstacles_y = y1[-self.dynamic_obstacle_count:]
        
        # 保存环境信息
        if save_data: 
            # 保存导航点，静态和动态障碍物坐标
            int_points_types = np.full(int_point_x.shape, 'IntPoints')
            static_types = np.full(static_obstacles_x.shape, 'Static')
            dynamic_types = np.full(dynamic_obstacles_x.shape, 'Dynamic')
            combined_data = np.column_stack((
                np.concatenate((int_point_x, static_obstacles_x, dynamic_obstacles_x)),
                np.concatenate((int_point_y, static_obstacles_y, dynamic_obstacles_y)),
                np.concatenate((int_points_types, static_types, dynamic_types))
            ))
            np.savetxt('points_positions.csv', combined_data, delimiter=',', fmt='%s', header='PosX,PosY,Type', comments='')
            
            # 保存智能体轨迹坐标
            with open('agents_info.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['PosX', 'PosY', 'Orientations'])
                for pos_x, pos_y, ori in zip(pos_x_list, pos_y_list, ori__list):
                    writer.writerow([pos_x, pos_y, ori])
                    
            # 保存环境其他信息
            with open('env_info.txt', 'w') as file:
                file.write("静态障碍物数量:"+str(self.static_obstacle_count) + '\n')
                file.write("动态障碍物数量:"+str(self.dynamic_obstacle_count) + '\n')
                file.write("障碍物尺寸:"+str(self.obstacle_radius) + '\n')
                file.write("智能体数量:"+str(self.n_agents) + '\n')
                file.write("导航点数量:"+str(self.int_points_num) + '\n')
            print("环境信息已保存！")
        
        sc1 = ax.scatter(int_point_x, int_point_y, c = 'r',marker = 'o') 
        hist_len = 8
        hist_sc = []
        # 历史轨迹散点图
        # for i in range(hist_len):
        #     temp_sc = ax.scatter(int_point_x, int_point_y, alpha=0.9, c='blue', marker = 'o')
        #     hist_sc.append(temp_sc)
        # 历史轨迹折线图
        x_vals, y_vals = [], []
        for i in range(hist_len):
            x_vals.append(int_point_x)
            y_vals.append(int_point_y)
        hist_line = ax.plot(x_vals, y_vals, alpha=0.8, color='blue', linestyle='--', marker='o', linewidth=0.1, markersize=0.1)
        hist_sc.append(hist_line)
        
        # 绘制静态障碍物的圆
        for x, y in zip(static_obstacles_x, static_obstacles_y):
            circle = Circle((x, y), self.obstacle_radius, color='pink', fill=True)
            ax.add_patch(circle)

        # 绘制动态障碍物的圆
        for x, y in zip(dynamic_obstacles_x, dynamic_obstacles_y):
            circle = Circle((x, y), self.obstacle_radius, color='green', fill=True)
            ax.add_patch(circle)

        # 设置x和y轴范围
        padding = 5  # 调整范围的padding大小
        x_min = min(np.min(pos_x_list), np.min(eva_pos_x_list)) - padding
        y_min = min(np.min(pos_y_list), np.min(eva_pos_y_list)) - padding
        x_max = max(np.max(pos_x_list), np.max(eva_pos_x_list)) + padding
        y_max = max(np.max(pos_y_list), np.max(eva_pos_y_list)) + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 更新函数，每次调用都会更新图表
        def update(frame):
            show_index = frame % len(pos_x_list)  # 使用取余运算防止越界
            x = pos_x_list[show_index]
            y = pos_y_list[show_index]
            x1 = eva_pos_x_list[show_index]
            y1 = eva_pos_y_list[show_index]

            angles = ori__list[show_index]
            show_index += 1
            dx = arrow_length * np.cos(angles)
            dy = arrow_length * np.sin(angles)

            # 清除旧的历史轨迹散点图
            for hist in hist_sc:
                for h in hist:
                    h.remove()
            hist_sc.clear()
            
            # start_index = max(0, show_index - hist_len)
            start_index = 0
            # 绘制历史轨迹散点图
            # for past_index in range(start_index, show_index):
            #     temp_sc = ax.scatter(pos_x_list[past_index], pos_y_list[past_index], alpha=0.3, c='blue', marker='o')
            #     hist_sc.append(temp_sc)
            
            # 绘制历史轨迹折线图
            x_vals, y_vals = [], []
            for past_index in range(start_index, show_index):
                x_vals.append(pos_x_list[past_index])
                y_vals.append(pos_y_list[past_index])
            hist_line = ax.plot(x_vals, y_vals, alpha=0.8, color='blue', linestyle='--', marker='o', linewidth=0.1, markersize=0.1)
            hist_sc.append(hist_line)
            
            sc.set_offsets(np.c_[x, y])
            arrows.set_offsets(np.c_[x, y])
            arrows.set_UVC(dx, dy)
            sc1.set_offsets(np.c_[x1, y1])
            # 在这里判断是否为最后一帧，并保存图片
            if frame == len(pos_x_list) - 1:
                plt.savefig('./navigation_final_frame.png')
                
        ani = animation.FuncAnimation(fig, update, frames=range(len(pos_x_list)), interval=100)
        ani.save('./navigation_animation_v2.gif', writer='pillow')  # 保存为GIF文件
        # plt.show()
        
