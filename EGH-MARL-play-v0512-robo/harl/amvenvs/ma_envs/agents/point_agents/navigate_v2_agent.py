import numpy as np
import fast_histogram as fh
from gym import spaces
from harl.envs.ma_envs.base import Agent
from harl.envs.ma_envs.commons import utils as U


class NavigateV2Agent(Agent):
    def __init__(self, experiment):
        super(NavigateV2Agent, self).__init__()
        self.local_mode = experiment.local_mode
        self.obs_radius = experiment.obs_radius
        self.comm_radius = experiment.comm_radius
        self.obs_radius2 = experiment.comm_radius / 2
        self.obs_mode = experiment.obs_mode
        self.distance_bins = experiment.distance_bins
        self.bearing_bins = experiment.bearing_bins
        self.torus = experiment.torus
        self.n_agents = experiment.nr_agents
        self.static_obstacle_count = experiment.static_obstacle_count
        self.dynamic_obstacle_count = experiment.dynamic_obstacle_count
        self.int_points_num = experiment.int_points_num
        self.landmarks_num = self.static_obstacle_count + self.dynamic_obstacle_count + self.int_points_num
        self.world_size = experiment.world_size
        self.dim_int_points_o = (1, 3)
        self.dim_obstacles_o = (self.static_obstacle_count + self.dynamic_obstacle_count, 3)
        self.dim_a = 2
        self.target = 0

        if self.obs_mode == '2d_rbf':
            self.dim_local_o = int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
            mu_d_e = np.linspace(0, self.world_size * np.sqrt(2) / 2, self.distance_bins)  # works for torus world, times 2 for non torus
            mu_d_n = np.linspace(0, self.world_size * np.sqrt(2) / 2, self.distance_bins)  # works for torus world, times 2 for non torus
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d_e = 4 * self.obs_radius2 / 80
            s_d_n = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d_e, mu_b)
            self.mu_e = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_e = np.hstack([s_d_e, s_b])
            xv, yv = np.meshgrid(mu_d_n, mu_b)
            self.mu_n = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_n = np.hstack([s_d_n, s_b])
        elif self.obs_mode == '2d_rbf_short':
            self.dim_local_o = int(not self.torus) + 3
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_int_points_o + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
            mu_d_e = np.linspace(0, self.world_size * np.sqrt(2) / 2, self.distance_bins)
            mu_d_n = np.linspace(0, self.world_size * np.sqrt(2) / 2, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d_e = 4 * self.obs_radius2 / 80
            s_d_n = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d_e, mu_b)
            self.mu_e = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_e = np.hstack([s_d_e, s_b])
            xv, yv = np.meshgrid(mu_d_n, mu_b)
            self.mu_n = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_n = np.hstack([s_d_n, s_b])
        elif self.obs_mode == '2d_rbf_limited':
            self.dim_local_o = int(not self.torus) + 1
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_int_points_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
            mu_d_e = np.linspace(0, self.obs_radius2, self.distance_bins)
            mu_d_n = np.linspace(0, self.comm_radius, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d_e = 4 * self.obs_radius2 / 80
            s_d_n = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d_e, mu_b)
            self.mu_e = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_e = np.hstack([s_d_e, s_b])
            xv, yv = np.meshgrid(mu_d_n, mu_b)
            self.mu_n = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_n = np.hstack([s_d_n, s_b])
        elif self.obs_mode == '2d_rbf_limited_short':
            self.dim_local_o = int(not self.torus) + 1
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_int_points_o = 3
            self.dim_flat_o = self.dim_int_points_o + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
            mu_d_e = np.linspace(0, self.obs_radius2, self.distance_bins)
            mu_d_n = np.linspace(0, self.comm_radius, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d_e = 4 * self.obs_radius2 / 80
            s_d_n = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d_e, mu_b)
            self.mu_e = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_e = np.hstack([s_d_e, s_b])
            xv, yv = np.meshgrid(mu_d_n, mu_b)
            self.mu_n = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s_n = np.hstack([s_d_n, s_b])
        elif self.obs_mode == 'sum_obs':
            self.dim_rec_o = (100, 7)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_int_points_o = (self.n_int_point, 3)
            self.dim_local_o = int(not self.torus)
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_no_ori':
            self.dim_rec_o = (100, 5)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = int(not self.torus)
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_limited':
            self.dim_int_points_o = (self.n_int_points, 3)
            self.dim_local_o = int(not self.torus) + 1
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self.dim_rec_o = (100, 8)
            self.dim_mean_embs = self.dim_rec_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_multi':
            self.dim_local_o = int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self.dim_rec_o = (100, 7)
            self.dim_mean_embs = (self.dim_rec_o, ) + (self.dim_int_points_o, )
            self._dim_o = np.prod(self.dim_rec_o) + np.prod(self.dim_int_points_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_multi_limited':
            self.dim_local_o = int(not self.torus) + 1
            self.dim_flat_o = self.dim_local_o
            self.dim_rec_o = (100, 8)
            self.dim_mean_embs = (self.dim_rec_o, ) + (self.dim_int_points_o, )
            self._dim_o = np.prod(self.dim_rec_o) + np.prod(self.dim_int_points_o) + self.dim_flat_o
        elif self.obs_mode == '2d_hist':
            self.dim_rec_o = (self.bearing_bins, self.distance_bins)
            self.dim_local_o = int(not self.torus) + int(self.obs_radius2 <= 100)
            self.dim_flat_o = self.dim_local_o + np.prod(self.dim_int_points_o)
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == '2d_hist_short':
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_local_o = int(not self.torus) + int(self.obs_radius2 <= 100) + 3
            self.dim_flat_o = self.dim_int_points_o + self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == 'fix':
            self.dim_rec_o = (self.n_agents - 1, 5 + int(self.obs_radius2 <= 100))
            self.dim_local_o = int(not self.torus) + int(self.obs_radius2 <= 100)
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + np.prod(self.dim_int_points_o) + self.dim_local_o
        elif self.obs_mode == 'eghn_acc':
            self.dim_rec_o = (self.n_agents - 1, 3 + int(self.obs_radius2 <= 100))
            self.dim_equ_o = 4
            self.dim_local_o = 0
            self.dim_flat_o = np.prod(self.dim_int_points_o) + np.prod(self.dim_obstacles_o) + self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_local_o  + self.dim_flat_o + self.dim_equ_o
            if self.local_mode:
                self._dim_o = self._dim_o + self.n_agents
        elif self.obs_mode == 'global':
            self.dim_rec_o = (self.n_agents - 1, 3)
            self.dim_equ_o = 4
            self.dim_local_o = 3 * int(not self.torus)
            self.dim_int_points_o = (1, 3)
            self.dim_flat_o = np.prod(self.dim_int_points_o) + np.prod(self.dim_obstacles_o) + self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o + self.dim_equ_o
            if self.local_mode:
                self._dim_o = self._dim_o + self.n_agents
        elif self.obs_mode == 'eghn':
            self.dim_rec_o = 0
            self.dim_local_o = 3 * int(not self.torus)
            self.dim_equ_o = 4
            self.dim_inv_o = 1
            self.dim_flat_o = np.prod(self.dim_int_points_o) + self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + np.prod(self.dim_int_points_o) + self.dim_local_o + self.dim_equ_o + self.dim_inv_o
        self.r_matrix = None
        self.graph_feature = None
        self.see_int_points = None
        self.dynamics = experiment.dynamics
        self.max_lin_velocity = 10  # cm/s
        self.max_ang_velocity = 2 * np.pi

    @property
    def observation_space(self):
        ob_space = spaces.Box(low=0., high=1., shape=(self._dim_o,), dtype=np.float32)
        ob_space.dim_local_o = self.dim_local_o
        # ob_space.dim_flat_o = self.dim_flat_o
        ob_space.dim_rec_o = self.dim_rec_o
        ob_space.dim_mean_embs = self.dim_mean_embs
        return ob_space

    @property
    def action_space(self):
        return spaces.Box(low=-1., high=+1., shape=(self.dim_a,), dtype=np.float32)

    def reset(self, state):
        self.state.p_pos = state[0:2]
        self.state.p_orientation = state[2]
        self.state.p_vel = np.zeros(2)
        self.state.w_vel = np.zeros(2)
        self.graph_feature = np.inf
        self.see_int_points = 0
        if self.obs_mode == 'sum_obs_learn_comm':
            self.action.c = 0

    def get_observation(self, dm, my_orientation, their_orientation, feat, vels):
        int_points_dists = dm[-self.landmarks_num:-self.landmarks_num+self.int_points_num]
        int_points_bearings = my_orientation[-self.landmarks_num:-self.landmarks_num+self.int_points_num]
        static_obstacles_dists = dm[-self.landmarks_num+self.int_points_num:-self.landmarks_num+self.int_points_num+self.static_obstacle_count]
        static_obstacles_bearings = my_orientation[-self.landmarks_num+self.int_points_num:-self.landmarks_num+self.int_points_num+self.static_obstacle_count]
        dynamic_obstacles_dists = dm[-self.dynamic_obstacle_count:]
        dynamic_obstacles_bearings = my_orientation[-self.dynamic_obstacle_count:]
        
        navigator_dists = dm[:-self.landmarks_num]
        navigator_bearings = my_orientation[:-self.landmarks_num]

        if self.obs_mode == 'fix':
            local_obs = self.get_local_obs()
            if self.obs_radius2 > 100:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
                local_obs = np.zeros(self.dim_local_o)
            else:
                if int_points_dists < self.obs_radius2:
                    dist_to_int_points = int_points_dists / self.obs_radius2
                    angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
                else:
                    dist_to_int_points = 1.
                    angle_to_int_points = [0, 0]

                see_int_points = 1 if dist_to_int_points < 1 else 0
                self.see_int_points = see_int_points

                shortest_path_to_int_points = self.graph_feature / (5 * self.comm_radius)\
                    if self.graph_feature < (5 * self.comm_radius) else 1.

                local_obs[-1] = shortest_path_to_int_points

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            ind = np.where(dm == -1)[0][0]

            fix_obs = np.zeros(self.dim_rec_o)

            if self.obs_radius2 > 100:
                fix_obs[:, 0] = np.concatenate([navigator_dists[0:ind], navigator_dists[ind + 1:]]) / self.comm_radius
                fix_obs[:, 1] = np.cos(np.concatenate([navigator_bearings[0:ind],
                                                       navigator_bearings[ind + 1:]]))
                fix_obs[:, 2] = np.sin(np.concatenate([navigator_bearings[0:ind],
                                                       navigator_bearings[ind + 1:]]))
                fix_obs[:, 3] = np.cos(np.concatenate([their_orientation[0:ind],
                                                       their_orientation[ind + 1:]]))
                fix_obs[:, 4] = np.sin(np.concatenate([their_orientation[0:ind],
                                                       their_orientation[ind + 1:]]))

            else:
                in_range = (int_points_dists < self.comm_radius) & (0 < int_points_dists)
                dists_in_range = np.array(feat)[in_range]
                dists_in_range_capped = np.where(dists_in_range <= 5 * self.comm_radius,
                                                 dists_in_range / (5 * self.comm_radius),
                                                 1.)
                fix_obs[:, 0] = np.concatenate([navigator_dists[0:ind], navigator_dists[ind + 1:]]) / self.comm_radius
                fix_obs[:, 1] = np.cos(np.concatenate([navigator_bearings[0:ind],
                                                       navigator_bearings[ind + 1:]]))
                fix_obs[:, 2] = np.sin(np.concatenate([navigator_bearings[0:ind],
                                                       navigator_bearings[ind + 1:]]))
                fix_obs[:, 3] = np.cos(np.concatenate([their_orientation[0:ind],
                                                       their_orientation[ind + 1:]]))
                fix_obs[:, 4] = np.sin(np.concatenate([their_orientation[0:ind],
                                                       their_orientation[ind + 1:]]))
                fix_obs[:, 5] = dists_in_range_capped

            obs = np.hstack([fix_obs.flatten(), int_points_obs.flatten(), local_obs.flatten()])
            
        elif self.obs_mode == 'eghn_acc':
            if self.obs_radius2 > 100:
                dist_to_static_obstacles = static_obstacles_dists / self.obs_radius2
                angle_to_static_obstacless = [np.cos(static_obstacles_bearings), np.sin(static_obstacles_bearings)]
                dist_to_dynamic_obstacles = dynamic_obstacles_dists / self.obs_radius2
                angle_to_dynamic_obstacless = [np.cos(dynamic_obstacles_bearings), np.sin(dynamic_obstacles_bearings)]
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
                local_obs = np.zeros(self.dim_local_o)
            int_points_obs = np.zeros((self.int_points_num, 3))
            return_int_points_obs = np.zeros((1, 3))
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]
            return_int_points_obs = int_points_obs[self.target]
            
            static_obstacles_obs = np.zeros((self.static_obstacle_count, 3))
            static_obstacles_obs[:, 0] = dist_to_static_obstacles
            static_obstacles_obs[:, 1] = angle_to_static_obstacless[0]
            static_obstacles_obs[:, 2] = angle_to_static_obstacless[1]
            dynamic_obstacles_obs = np.zeros((self.dynamic_obstacle_count, 3))
            dynamic_obstacles_obs[:, 0] = dist_to_dynamic_obstacles
            dynamic_obstacles_obs[:, 1] = angle_to_dynamic_obstacless[0]
            dynamic_obstacles_obs[:, 2] = angle_to_dynamic_obstacless[1]
                
            ind = np.where(dm == -1)[0][0]

            equ_obs = np.zeros(self.dim_equ_o)
            # add value normalization
            equ_obs[0] = self.state.p_pos[0] / self.world_size
            equ_obs[1] = self.state.p_pos[1] / self.world_size
            equ_obs[2] = self.state.p_vel[0] / self.max_lin_velocity
            equ_obs[3] = self.state.p_vel[1] / self.max_lin_velocity
            
            eghn_obs = np.zeros(self.dim_rec_o)
            eghn_obs[:, 0] = np.concatenate([navigator_dists[0:ind], navigator_dists[ind + 1:]]) / self.comm_radius
            eghn_obs[:, 1] = np.cos(np.concatenate([navigator_bearings[0:ind], navigator_bearings[ind + 1:]]))
            eghn_obs[:, 2] = np.sin(np.concatenate([navigator_bearings[0:ind], navigator_bearings[ind + 1:]]))
            if self.local_mode:
                local_distances = np.concatenate([navigator_dists[0:ind], navigator_dists[ind + 1:]]) / self.comm_radius
                adjacency_vector = np.ones(self.n_agents) * -1  # 初始化为-1，表示没有连接
                adjusted_distances = np.insert(local_distances, ind, -1)  # 在当前智能体位置插入一个占位值
                for i, distance in enumerate(adjusted_distances):
                    if i != ind and distance <= self.obs_radius / self.comm_radius:
                        adjacency_vector[i] = 1  # 如果智能体间距离小于阈值，则设置为1
                obs = np.hstack([eghn_obs.flatten(), return_int_points_obs.flatten(), \
                                 static_obstacles_obs.flatten(), dynamic_obstacles_obs.flatten(), \
                                 equ_obs.flatten(), adjacency_vector])
            else:
                obs = np.hstack([eghn_obs.flatten(), return_int_points_obs.flatten(), 
                                 static_obstacles_obs.flatten(), dynamic_obstacles_obs.flatten(),
                                 equ_obs.flatten()])

        elif self.obs_mode == 'global':
            ind = np.where(dm == -1)[0][0]
            
            # 兴趣点观测
            if int_points_dists[self.target] <= self.obs_radius:
                dist_to_int_points = int_points_dists[self.target] / self.obs_radius
                angle_to_int_points = [np.cos(int_points_bearings[self.target]), np.sin(int_points_bearings[self.target])]
            else:
                dist_to_int_points = 1.
                angle_to_int_points = [0., 0.]
            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            # 静态障碍物观测
            static_obstacles_obs = np.zeros((self.static_obstacle_count, 3))
            static_obstacles_obs[:, 0] = np.where(static_obstacles_dists <= self.obs_radius, static_obstacles_dists, 1.)
            static_obstacles_obs[:, 1] = np.where(static_obstacles_dists <= self.obs_radius, np.sin(static_obstacles_bearings), 0.)
            static_obstacles_obs[:, 2] = np.where(static_obstacles_dists <= self.obs_radius, np.cos(static_obstacles_bearings), 0.)

            # 动态障碍物观测
            dynamic_obstacles_obs = np.zeros((self.dynamic_obstacle_count, 3))
            dynamic_obstacles_obs[:, 0] = np.where(dynamic_obstacles_dists <= self.obs_radius, dynamic_obstacles_dists, 1.)
            dynamic_obstacles_obs[:, 1] = np.where(dynamic_obstacles_dists <= self.obs_radius, np.cos(dynamic_obstacles_bearings), 0.)
            dynamic_obstacles_obs[:, 2] = np.where(dynamic_obstacles_dists <= self.obs_radius, np.sin(dynamic_obstacles_bearings), 0.)

            # 局部观测
            local_obs = np.zeros(self.dim_local_o)
            local_obs = self.get_local_obs()

            # 等变观测
            equ_obs =  np.zeros(self.dim_equ_o)
            # add value normalization
            equ_obs[0] = self.state.p_pos[0] / self.world_size
            equ_obs[1] = self.state.p_pos[1] / self.world_size
            equ_obs[2] = self.state.p_vel[0] / self.max_lin_velocity
            equ_obs[3] = self.state.p_vel[1] / self.max_lin_velocity

            # 其余智能体的观测
            rec_obs = np.zeros(self.dim_rec_o)
            dis_to_other = np.concatenate([navigator_dists[0:ind], navigator_dists[ind + 1:]])
            orientation = np.concatenate([navigator_bearings[0:ind], navigator_bearings[ind + 1:]])
            rec_obs = np.zeros(self.dim_rec_o)
            rec_obs[:, 0] = np.where(dis_to_other <= self.obs_radius, dis_to_other / self.obs_radius, 1)  # 通过obs_raadius归一化，超过的值置1
            rec_obs[:, 1] = np.where(dis_to_other <= self.obs_radius, np.cos(orientation), 0)  # 当距离超过obs_radius，置0
            rec_obs[:, 2] = np.where(dis_to_other <= self.obs_radius, np.sin(orientation), 0)  # 当距离超过obs_radius，置0

            if self.local_mode:
                adjacency_vector = np.where(navigator_dists <= self.obs_radius, 1, -1)
                adjacency_vector[ind] = -1
                obs = np.hstack([rec_obs.flatten(), int_points_obs.flatten(), static_obstacles_obs.flatten(), \
                                 dynamic_obstacles_obs.flatten(), local_obs.flatten(), equ_obs.flatten(), adjacency_vector])
            else:
                obs = np.hstack([rec_obs.flatten(), int_points_obs.flatten(), static_obstacles_obs.flatten(), \
                                 dynamic_obstacles_obs.flatten(), local_obs.flatten(), equ_obs.flatten()])

        elif self.obs_mode == 'eghn':
            dist_to_int_points = int_points_dists / self.obs_radius2
            angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
            
            local_obs = np.zeros(self.dim_local_o)
            local_obs = self.get_local_obs()

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            equ_obs = np.zeros(self.dim_equ_o)
            # add value normalization
            equ_obs[0] = self.state.p_pos[0] / self.world_size
            equ_obs[1] = self.state.p_pos[1] / self.world_size
            equ_obs[2] = self.state.p_vel[0] / self.max_lin_velocity
            equ_obs[3] = self.state.p_vel[1] / self.max_lin_velocity
            
            inv_obs = np.zeros(self.dim_inv_o)
            inv_obs[0] = np.linalg.norm(self.state.p_vel) / self.max_lin_velocity
            obs = np.hstack([inv_obs.flatten(), local_obs.flatten(), int_points_obs.flatten(), equ_obs.flatten()])
        elif self.obs_mode == 'sum_obs':
            # local obs
            if int_points_dists < self.obs_radius2:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
            else:
                dist_to_int_points = 1.
                angle_to_int_points = [0, 0]

            local_obs = np.zeros(self.dim_local_o)

            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                    wall = 1
                else:
                    wall = 0
                local_obs[0] = wall

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            # neighbor obs
            navigators_in_range = (navigator_dists < self.comm_radius) & (0 < navigator_dists)
            nr_neighbors = np.sum(navigators_in_range)

            sum_obs = np.zeros(self.dim_rec_o)

            nr_agents = dm.size - 1

            sum_obs[0:nr_neighbors, 0] = navigator_dists[navigators_in_range] / self.comm_radius
            sum_obs[0:nr_neighbors, 1] = np.cos(navigator_bearings[navigators_in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(navigator_bearings[navigators_in_range])
            sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[navigators_in_range])
            sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[navigators_in_range])
            sum_obs[0:nr_neighbors, 5] = 1
            sum_obs[0:nr_agents, 6] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs.flatten(), int_points_obs.flatten()])

        elif self.obs_mode == 'sum_obs_no_ori':
            # local obs
            if int_points_dists < self.obs_radius2:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
            else:
                dist_to_int_points = 1.
                angle_to_int_points = [0, 0]

            local_obs = np.zeros(self.dim_local_o)

            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                    wall = 1
                else:
                    wall = 0
                local_obs[0] = wall

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            # neighbor obs
            navigators_in_range = (navigator_dists < self.comm_radius) & (0 < navigator_dists)
            nr_neighbors = np.sum(navigators_in_range)

            sum_obs = np.zeros(self.dim_rec_o)

            nr_agents = dm.size - 1

            sum_obs[0:nr_neighbors, 0] = navigator_dists[navigators_in_range] / self.comm_radius
            sum_obs[0:nr_neighbors, 1] = np.cos(navigator_bearings[navigators_in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(navigator_bearings[navigators_in_range])
            sum_obs[0:nr_neighbors, 3] = 1
            sum_obs[0:nr_agents, 4] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs.flatten(), int_points_obs.flatten()])

        elif self.obs_mode == 'sum_obs_limited':
            if int_points_dists < self.obs_radius2:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
            else:
                dist_to_int_points = 1.
                angle_to_int_points = [0, 0]

            see_int_points = 1 if dist_to_int_points < 1 else 0
            self.see_int_points = see_int_points

            shortest_path_to_int_points = self.graph_feature / (5 * self.comm_radius)\
                if self.graph_feature < (5 * self.comm_radius) else 1.

            local_obs = np.zeros(self.dim_local_o)
            local_obs[0] = shortest_path_to_int_points

            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                    wall = 1
                else:
                    wall = 0
                local_obs[1] = wall

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            # neighbor obs
            int_points_in_range = (int_points_dists < self.obs_radius2) & (0 < int_points_dists)
            navigators_in_range = (navigator_dists < self.comm_radius) & (0 < navigator_dists)
            nr_neighbors = np.sum(navigators_in_range)

            dists_in_range = np.array(feat)[navigators_in_range]
            dists_in_range_capped = np.where(dists_in_range <= 5 * self.comm_radius,
                                             dists_in_range / (5 * self.comm_radius),
                                             1.)

            sum_obs = np.zeros(self.dim_rec_o)

            nr_agents = dm.size - 1

            sum_obs[0:nr_neighbors, 0] = navigator_dists[navigators_in_range] / self.comm_radius
            sum_obs[0:nr_neighbors, 1] = np.cos(navigator_bearings[navigators_in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(navigator_bearings[navigators_in_range])
            sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[navigators_in_range])
            sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[navigators_in_range])
            sum_obs[0:nr_neighbors, 5] = dists_in_range_capped
            sum_obs[0:nr_neighbors, 6] = 1
            sum_obs[0:nr_agents, 7] = 1

                # obs = np.hstack([sum_obs.flatten(), local_obs])
            obs = np.hstack([sum_obs.flatten(), local_obs.flatten(), int_points_obs.flatten()])

        elif self.obs_mode == 'sum_obs_multi':
            in_range = (int_points_dists <= self.obs_radius2) & (0 <= int_points_dists)
            nr_neighboring_int_points = np.sum(in_range)
            dist_to_int_points = int_points_dists[in_range] / self.obs_radius2
            angle_to_int_points = [np.cos(int_points_bearings[in_range]),
                               np.sin(int_points_bearings[in_range])]

            if self.obs_radius2 > 100:
                local_obs = np.zeros(self.dim_local_o)

                if self.torus is False:
                    if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                        wall = 1
                    else:
                        wall = 0
                    local_obs[0] = wall
            else:
                shortest_path_to_int_points = self.graph_feature / (5 * self.comm_radius)\
                    if self.graph_feature < (5 * self.comm_radius) else 1.

                local_obs = np.zeros(self.dim_local_o)
                local_obs[0] = shortest_path_to_int_points

                if self.torus is False:
                    if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                        wall = 1
                    else:
                        wall = 0
                    local_obs[1] = wall

            sum_int_points_obs = np.zeros(self.dim_int_points_o)
            sum_int_points_obs[:nr_neighboring_int_points, 0] = dist_to_int_points
            sum_int_points_obs[:nr_neighboring_int_points, 1] = angle_to_int_points[0]
            sum_int_points_obs[:nr_neighboring_int_points, 2] = angle_to_int_points[1]
            sum_int_points_obs[:nr_neighboring_int_points, 3] = 1
            sum_int_points_obs[:self.n_int_points, 4] = 1

            navigators_in_range = (navigator_dists <= self.comm_radius) & (0 <= navigator_dists)
            nr_neighbors = np.sum(navigators_in_range)

            if self.obs_radius2 > 100:
                sum_obs = np.zeros(self.dim_rec_o)

                nr_agents = dm.size - self.n_int_points

                sum_obs[0:nr_neighbors, 0] = navigator_dists[navigators_in_range] / self.comm_radius
                sum_obs[0:nr_neighbors, 1] = np.cos(navigator_bearings[navigators_in_range])
                sum_obs[0:nr_neighbors, 2] = np.sin(navigator_bearings[navigators_in_range])
                sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[navigators_in_range])
                sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[navigators_in_range])
                sum_obs[0:nr_neighbors, 5] = 1
                sum_obs[0:nr_agents, 6] = 1

            else:
                dists_in_range = np.array(feat)[navigators_in_range]
                dists_in_range_capped = np.where(dists_in_range <= 5 * self.comm_radius,
                                                 dists_in_range / (5 * self.comm_radius),
                                                 1.)

                sum_obs = np.zeros(self.dim_rec_o)

                nr_agents = dm.size - self.n_int_points

                sum_obs[0:nr_neighbors, 0] = navigator_dists[navigators_in_range] / self.comm_radius
                sum_obs[0:nr_neighbors, 1] = np.cos(navigator_bearings[navigators_in_range])
                sum_obs[0:nr_neighbors, 2] = np.sin(navigator_bearings[navigators_in_range])
                sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[navigators_in_range])
                sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[navigators_in_range])
                sum_obs[0:nr_neighbors, 5] = dists_in_range_capped
                sum_obs[0:nr_neighbors, 6] = 1
                sum_obs[0:nr_agents, 7] = 1

            obs = np.hstack([sum_obs.flatten(), sum_int_points_obs.flatten(), local_obs])

        elif self.obs_mode == '2d_hist':
            if self.obs_radius2 > 100:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]

                local_obs = np.zeros(self.dim_flat_o)

                if self.torus is False:
                    if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                        wall = 1
                    else:
                        wall = 0
                    local_obs[0] = wall
            else:
                if int_points_dists < self.obs_radius2:
                    dist_to_int_points = int_points_dists / self.obs_radius2
                    angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
                else:
                    dist_to_int_points = 1.
                    angle_to_int_points = [0, 0]

                see_int_points = 1 if dist_to_int_points < 1 else 0
                self.see_int_points = see_int_points

                shortest_path_to_int_points = self.graph_feature / (5 * self.comm_radius)\
                    if self.graph_feature < (5 * self.comm_radius) else 1.

                local_obs = np.zeros(self.dim_flat_o)
                local_obs[0] = shortest_path_to_int_points

                if self.torus is False:
                    if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                        wall = 1
                    else:
                        wall = 0
                    local_obs[1] = wall

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[:, 0] = dist_to_int_points
            int_points_obs[:, 1] = angle_to_int_points[0]
            int_points_obs[:, 2] = angle_to_int_points[1]

            # neighbor obs
            navigators_in_range = (navigator_dists < self.comm_radius) & (0 < navigator_bearings)
            int_points_in_range = (int_points_dists < self.obs_radius2) & (0 < int_points_dists)
            nr_agents = dm.size - 2  # exclude self and int_points

            hist_2d_agents = fh.histogram2d(navigator_bearings[navigators_in_range], navigator_dists[navigators_in_range],
                                           bins=(self.bearing_bins, self.distance_bins),
                                           range=[[-np.pi, np.pi], [0, self.world_size * np.sqrt(2) / 2]])
            hist_2d_int_points = fh.histogram2d(int_points_bearings[int_points_in_range], int_points_dists[int_points_in_range],
                                            bins=(self.bearing_bins, self.distance_bins),
                                            range=[[-np.pi, np.pi], [0, self.world_size * np.sqrt(2) / 2]])
            histogram = np.hstack([hist_2d_agents.flatten() / (nr_agents),
                                   hist_2d_int_points.flatten()])

            obs = np.hstack([histogram, local_obs])

        elif self.obs_mode == '2d_hist_short':
            if self.obs_radius2 > 100:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]

            else:
                if int_points_dists < self.obs_radius2:
                    dist_to_int_points = int_points_dists / self.obs_radius2
                    angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
                else:
                    dist_to_int_points = 1.
                    angle_to_int_points = [0, 0]

                see_int_points = 1 if dist_to_int_points < 1 else 0
                self.see_int_points = see_int_points

            local_obs = self.get_local_obs()

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[0] = dist_to_int_points
            int_points_obs[1] = angle_to_int_points[0]
            int_points_obs[2] = angle_to_int_points[1]

            # neighbor obs
            in_range = (navigator_dists < self.comm_radius) & (0 < navigator_dists)
            nr_agents = dm.size - 2  # exclude self and int_points

            hist_2d_agents = fh.histogram2d(navigator_bearings[in_range], navigator_dists[in_range],
                                            bins=(self.bearing_bins, self.distance_bins),
                                            range=[[-np.pi, np.pi], [0, self.world_size * np.sqrt(2) / 2]])
            histogram = hist_2d_agents.flatten() / (nr_agents - 1)

            obs = np.hstack([histogram, int_points_obs.flatten(), local_obs])

        elif self.obs_mode == '2d_rbf':
            local_obs = np.zeros(self.dim_local_o)

            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                    wall = 1
                else:
                    wall = 0
                local_obs[0] = wall
            int_points_in_range = int_points_dists < self.obs_radius2
            if np.any(int_points_in_range):
                dbn = np.stack([int_points_dists[int_points_in_range], int_points_bearings[int_points_in_range] + np.pi], axis=1)
                int_points_rbf_hist = U.get_weights_2d(dbn, self.mu_e, self.s_e, [self.bearing_bins, self.distance_bins])

            else:
                int_points_rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            in_range = (0 < navigator_dists) & (navigator_dists < self.comm_radius)

            if np.any(in_range):
                dbn = np.stack([navigator_dists[in_range], navigator_bearings[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu_n, self.s_n, [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, int_points_rbf_hist.flatten(), local_obs])

        elif self.obs_mode == '2d_rbf_short':
            dist_to_int_points = int_points_dists / self.obs_radius2
            angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]

            see_int_points = 1 if dist_to_int_points < 1 else 0
            self.see_int_points = see_int_points

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[0] = dist_to_int_points
            int_points_obs[1] = angle_to_int_points[0]
            int_points_obs[2] = angle_to_int_points[1]

            local_obs = self.get_local_obs()

            in_range = (0 < navigator_dists) & (navigator_dists < self.comm_radius)

            if np.any(in_range):
                dbn = np.stack([navigator_dists[in_range], navigator_bearings[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu_n, self.s_n, [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, int_points_obs, local_obs])

        elif self.obs_mode == '2d_rbf_limited':
            if int_points_dists < self.obs_radius2:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = [np.cos(int_points_bearings), np.sin(int_points_bearings)]
            else:
                dist_to_int_points = 1.
                angle_to_int_points = [0, 0]

            see_int_points = 1 if dist_to_int_points < 1 else 0
            self.see_int_points = see_int_points

            shortest_path_to_int_points = self.graph_feature / (5 * self.comm_radius)\
                if self.graph_feature < (5 * self.comm_radius) else 1.

            local_obs = np.zeros(self.dim_flat_o)
            local_obs[0] = shortest_path_to_int_points

            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                    wall = 1
                else:
                    wall = 0
                local_obs[1] = wall
            int_points_in_range = int_points_dists < self.obs_radius2
            sub = []
            if int_points_in_range:
                dbn = np.hstack([int_points_dists, int_points_bearings + np.pi])
                int_points_rbf_hist = U.get_weights_2d(dbn, self.mu_e, self.s_e, [self.bearing_bins, self.distance_bins])

            else:
                int_points_rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            in_range = (0 < navigator_dists) & (navigator_dists < self.comm_radius)

            if np.any(in_range):
                dbn = np.stack([navigator_dists[in_range], navigator_bearings[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu_n, self.s_n, [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, int_points_rbf_hist.flatten(), local_obs])

        elif self.obs_mode == '2d_rbf_limited_short':
            if int_points_dists < self.obs_radius2:
                dist_to_int_points = int_points_dists / self.obs_radius2
                angle_to_int_points = np.array([np.cos(int_points_bearings), np.sin(int_points_bearings)])
            else:
                dist_to_int_points = 1.
                angle_to_int_points = [0, 0]

            see_int_points = 1 if dist_to_int_points < 1 else 0
            self.see_int_points = see_int_points

            shortest_path_to_int_points = self.graph_feature / (5 * self.comm_radius)\
                if self.graph_feature < (5 * self.comm_radius) else 1.

            int_points_obs = np.zeros(self.dim_int_points_o)
            int_points_obs[0] = dist_to_int_points
            int_points_obs[1] = angle_to_int_points[0]
            int_points_obs[2] = angle_to_int_points[1]

            local_obs = np.zeros(self.dim_local_o)
            local_obs[0] = shortest_path_to_int_points

            if self.torus is False:
                if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
                    wall = 1
                else:
                    wall = 0
                local_obs[1] = wall

            in_range = (0 < navigator_dists) & (navigator_dists < self.comm_radius)

            if np.any(in_range):
                dbn = np.stack([navigator_dists[in_range], navigator_bearings[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu_n, self.s_n,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, int_points_obs, local_obs])

        return obs

    def set_position(self, x_2):
        assert x_2.shape == (2,)
        self.position = x_2

    def set_angle(self, phi):
        assert phi.shape == (1,)
        self.angle = phi
        r_matrix_1 = np.squeeze([[np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
        r_matrix_2 = np.squeeze([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

        self.r_matrix = np.dot(r_matrix_1, r_matrix_2)

    def get_local_obs_acc(self):
        local_obs = np.zeros(self.dim_local_o)
        local_obs[0] = self.state.p_vel[0] / self.max_lin_velocity
        local_obs[1] = self.state.p_vel[1] / self.max_ang_velocity

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
                wall_dists = np.array([self.world_size - self.state.p_pos[0],
                                       self.world_size - self.state.p_pos[1],
                                       self.state.p_pos[0],
                                       self.state.p_pos[1]])
                wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
                closest_wall = np.argmin(wall_dists)
                local_obs[2] = wall_dists[closest_wall]
                local_obs[3] = np.cos(wall_angles[closest_wall])
                local_obs[4] = np.sin(wall_angles[closest_wall])
            else:
                local_obs[2] = 1
                local_obs[3:5] = 0

        return local_obs

    def get_local_obs(self):
        local_obs = np.zeros(self.dim_local_o)

        if self.torus is False:
            if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
                wall_dists = np.array([self.world_size - self.state.p_pos[0],
                                       self.world_size - self.state.p_pos[1],
                                       self.state.p_pos[0],
                                       self.state.p_pos[1]])
                wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
                closest_wall = np.argmin(wall_dists)
                local_obs[0] = wall_dists[closest_wall]

                local_obs[1] = np.cos(wall_angles[closest_wall])
                local_obs[2] = np.sin(wall_angles[closest_wall])
            else:
                local_obs[0] = 1
                local_obs[1:3] = 0

        return local_obs

