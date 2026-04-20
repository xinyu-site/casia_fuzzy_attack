from harl.envs.ma_envs.base import Agent
import harl.envs.ma_envs.commons.utils as U
from gym import spaces
import numpy as np
import fast_histogram as fh
import torch


class PointAgent(Agent):
    def __init__(self, experiment):
        super(PointAgent, self).__init__()
        self.local_mode = experiment.local_mode
        self.obs_radius = experiment.obs_radius
        self.comm_radius = experiment.comm_radius
        self.obs_mode = experiment.obs_mode
        self.distance_bins = experiment.distance_bins
        self.bearing_bins = experiment.bearing_bins
        self.torus = experiment.torus
        self.n_agents = experiment.nr_agents
        self.world_size = experiment.world_size
        self._dim_a = 2
        self.dim_local_o = 3 + int(not self.torus)
        self.num_features = 5
        # 计算各种观察模式的维度
        if self.obs_mode == 'gcn':
            self.ob_space = self.num_features
            print(self.obs_mode)
            print(self.num_features)
        elif self.obs_mode == '2d_rbf_acc':
            mu_d = np.linspace(0, self.world_size * np.sqrt(2), self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b])
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '3d_rbf':
            mu_d = np.linspace(0, self.world_size * np.sqrt(2), self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv, zv = np.meshgrid(mu_d, mu_b, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b, s_b])
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '2d_rbf_acc_limited':
            mu_d = np.linspace(0, self.comm_radius, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b])
            self.dim_local_o = 3 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '2d_rbf_limited':
            mu_d = np.linspace(0, self.comm_radius, self.distance_bins)
            mu_b = np.linspace(0, 2 * np.pi, self.bearing_bins)
            s_d = 4 * self.comm_radius / 80
            s_b = 0.33
            xv, yv = np.meshgrid(mu_d, mu_b)
            self.mu = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.s = np.hstack([s_d, s_b])
            self.dim_local_o = 1 + 3 * int(not self.torus)
            self.dim_rec_o = (self.distance_bins, self.bearing_bins)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
            self.dim_mean_embs = None
        elif self.obs_mode == '2d_hist_acc':
            self.dim_rec_o = (self.bearing_bins, self.distance_bins)
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc':
            self.dim_rec_o = (self.n_agents - 1, 7)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc_full':
            self.dim_rec_o = (100, 9)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc_no_vel':
            self.dim_rec_o = (100, 5)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_acc_limited':
            self.dim_rec_o = (100, 8)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 3 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs':
            self.dim_rec_o = (100, 7)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 1 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'sum_obs_limited':
            self.dim_rec_o = (100, 8)
            self.dim_mean_embs = self.dim_rec_o
            self.dim_local_o = 1 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_flat_o
        elif self.obs_mode == 'fix_acc':
            self.dim_rec_o = (self.n_agents - 1, 5)
            self.dim_local_o = 2 + 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_local_o
        elif self.obs_mode == 'fix':
            self.dim_rec_o = (self.n_agents - 1, 3)
            self.dim_local_o = 3 * int(not self.torus)
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_local_o
        elif self.obs_mode == 'eghn_acc':
            self.dim_rec_o = (self.n_agents - 1, 3)
            self.dim_local_o = 0
            self.dim_equ_o = 4
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = self.dim_rec_o
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_local_o + self.dim_equ_o
            if self.local_mode:
                self._dim_o = self._dim_o + self.n_agents
        elif self.obs_mode == 'hepn_local':
            self.dim_rec_o = (self.n_agents - 1, 3)
            self.dim_local_o = 3 * int(not self.torus)
            self.dim_equ_o = 4
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = np.prod(self.dim_rec_o) + self.dim_local_o + self.dim_equ_o
            if self.local_mode:
                self._dim_o = self._dim_o + self.n_agents
        elif self.obs_mode == 'gnn_local':
            # self.dim_rec_o = (self.n_agents - 1, 3)
            self.dim_local_o = 3 * int(not self.torus)
            self.dim_equ_o = 4
            self.dim_flat_o = self.dim_local_o
            self.dim_mean_embs = None
            self._dim_o = self.dim_local_o + self.dim_equ_o
            if self.local_mode:
                self._dim_o = self._dim_o + self.n_agents
        else:
            raise ValueError('obs mode must be 1D or 2D')
        self.r_matrix = None
        self.feature = None
        self.complete = False
        self.n_sensors = 4
        self.sensor_range = 0.5
        self.radius = 0.2
        angles_K = np.linspace(0., 2. * np.pi, self.n_sensors + 1)[:-1]
        sensor_vecs_K_2 = np.c_[np.cos(angles_K), np.sin(angles_K)]

        self.sensors = sensor_vecs_K_2
        self.rel_vel_hist = []
        self.neighborhood_size_hist = []

    @property
    def observation_space(self):

        ob_space = spaces.Box(low=0., high=1., shape=(self._dim_o,), dtype=np.float32)
        ob_space.dim_local_o = self.dim_local_o
        ob_space.dim_flat_o = self.dim_flat_o
        ob_space.dim_rec_o = self.dim_rec_o
        ob_space.dim_mean_embs = self.dim_mean_embs

        '''
        ob_space = spaces.Box(low=0., high=1., shape=(self.num_features,), dtype=np.float32)
        '''
        return ob_space

    @property
    def action_space(self):
        return spaces.Box(np.array([-1., -1.]), np.array([1., 1.]), dtype=np.float32)

    def set_velocity(self, vel):
        self.velocity = vel

    def reset(self, state):
        self.state.p_pos = state[0:2]
        self.state.p_orientation = state[2]
        # self.state.p_vel = np.zeros(2)
        self.state.p_vel = np.random.uniform(-0.01, 0.01, 2)
        self.state.w_vel = np.zeros(2)
        self.feature = np.inf
        self.complete = False
        
    def set_ue_vel(self, vel):
        self.state.p_vel = vel

    def get_observation(self, dm, my_orientation, their_orientation, vels, nh_size, agent_id,nodes=None):
        # w_vel 是0，以自己为坐标系，求相对速度
        if self.obs_mode == 'fix_acc':
            ind = np.where(dm == -1)[0][0]
            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()

            fix_obs = np.zeros(self.dim_rec_o)

            fix_obs[:, 0] = np.concatenate([dm[0:ind], dm[ind + 1:]]) / self.comm_radius
            fix_obs[:, 1] = np.cos(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            fix_obs[:, 2] = np.sin(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            fix_obs[:, 3] = np.concatenate([rel_vels[0:ind, 0], rel_vels[ind + 1:, 0]]) / (2 * self.max_lin_velocity)
            fix_obs[:, 4] = np.concatenate([rel_vels[0:ind, 1], rel_vels[ind + 1:, 1]]) / (2 * self.max_lin_velocity)

            obs = np.hstack([fix_obs.flatten(), local_obs.flatten()])
        elif self.obs_mode == 'fix':
            ind = np.where(dm == -1)[0][0]
            local_obs = self.get_local_obs()
            fix_obs = np.zeros(self.dim_rec_o)
            fix_obs[:, 0] = np.concatenate([dm[0:ind], dm[ind + 1:]]) / self.comm_radius
            fix_obs[:, 1] = np.cos(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            fix_obs[:, 2] = np.sin(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            obs = np.hstack([fix_obs.flatten(), local_obs.flatten()])

        elif self.obs_mode == 'gcn':

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs()
            # local_obs = self.get_local_obs_acc()

            nh_size = np.sum((0 < dm) & (dm < self.comm_radius))
            #
            edge_index1 = np.ones((2, nh_size + 1), dtype=np.long) * agent_id
            j = 0
            for i, val in enumerate(dm < self.comm_radius):
                if val:
                    edge_index1[0][j] = i
                    j = j + 1

            edge_index2 = edge_index1[[1, 0], :]
            edge_index = np.concatenate((edge_index2, edge_index1), axis=1)

            # x=np.concatenate((dm,my_orientation,their_orientation,vels.reshape([20,])))
            # x=np.concatenate((dm.reshape([1,self.n_agents]),my_orientation.reshape([1,self.n_agents]),their_orientation.reshape([1,self.n_agents]),vels.reshape([1,self.n_agents*2])),axis=1)

            x = np.concatenate((dm.reshape([self.n_agents, 1]), my_orientation.reshape([self.n_agents, 1]),
                                their_orientation.reshape([self.n_agents, 1]), vels), axis=1)

            datax = torch.tensor(x, dtype=torch.float)
            edge = torch.tensor(edge_index, dtype=torch.long)

            # obs=[datax,edge]
            # obs=[edge_index]
            # obs=dm,my_orientation,their_orientation,vels,nh_size,local_obs,edge_index
            obs = datax, edge

        elif self.obs_mode == '2d_hist_acc':
            local_obs = self.get_local_obs_acc()
            in_range = (0 < dm) & (dm < self.comm_radius)
            hist_2d = fh.histogram2d(my_orientation[in_range], dm[in_range],
                                     bins=(self.bearing_bins, self.distance_bins),
                                     range=[[-np.pi, np.pi], [0, self.world_size * np.sqrt(2)]])
            histogram = hist_2d.flatten() / (self.n_agents - 1)
            obs = np.hstack([histogram, local_obs])

        elif self.obs_mode == '2d_rbf_acc':
            in_range = (dm < self.comm_radius) & (0 < dm)

            local_obs = self.get_local_obs_acc()

            if np.any(in_range):
                dbn = np.stack([dm[in_range], my_orientation[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == '3d_rbf':
            in_range = (dm < self.comm_radius) & (0 < dm)

            local_obs = self.get_local_obs_acc()

            if np.any(in_range):
                dbn = np.stack([dm[in_range],
                                my_orientation[in_range] + np.pi,
                                their_orientation[in_range] + np.pi],
                               axis=1)
                rbf_hist = U.get_weights_3d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins, self.bearing_bins]) / (
                                       self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins, self.bearing_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == '2d_rbf_acc_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs_acc()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            if np.any(in_range):
                dbn = np.stack([dm[in_range], my_orientation[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == '2d_rbf_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            if np.any(in_range):
                dbn = np.stack([dm[in_range], my_orientation[in_range] + np.pi], axis=1)
                rbf_hist = U.get_weights_2d(dbn, self.mu, self.s,
                                            [self.bearing_bins, self.distance_bins]) / (self.n_agents - 1)

            else:
                rbf_hist = np.zeros([self.bearing_bins, self.distance_bins])

            rbf_hist_flat = rbf_hist.flatten()

            obs = np.hstack([rbf_hist_flat, local_obs])

        elif self.obs_mode == 'sum_obs_acc':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[:nr_neighbors, 3] = rel_vels[:, 0][in_range] / (2 * self.max_lin_velocity)
            sum_obs[:nr_neighbors, 4] = rel_vels[:, 1][in_range] / (2 * self.max_lin_velocity)
            sum_obs[0:nr_neighbors, 5] = 1
            sum_obs[0:self.n_agents - 1, 6] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_acc_full':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[in_range])
            sum_obs[:nr_neighbors, 5] = rel_vels[:, 0][in_range] / (2 * self.max_lin_velocity)
            sum_obs[:nr_neighbors, 6] = rel_vels[:, 1][in_range] / (2 * self.max_lin_velocity)
            sum_obs[0:nr_neighbors, 7] = 1
            sum_obs[0:self.n_agents - 1, 8] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_acc_no_vel':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs_acc()

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = 1
            sum_obs[0:self.n_agents - 1, 4] = 1
            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_acc_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            rel_vels = self.state.w_vel - vels

            local_obs = self.get_local_obs_acc()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = (nh_size[in_range] - nr_neighbors) / (self.n_agents - 2) if self.n_agents > 2 \
                else np.zeros(nr_neighbors)
            sum_obs[:nr_neighbors, 4] = rel_vels[:, 0][in_range] / (2 * self.max_lin_velocity)
            sum_obs[:nr_neighbors, 5] = rel_vels[:, 1][in_range] / (2 * self.max_lin_velocity)
            sum_obs[0:nr_neighbors, 6] = 1
            sum_obs[0:self.n_agents - 1, 7] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = np.cos(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 4] = np.sin(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 5] = 1
            sum_obs[0:self.n_agents - 1, 6] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])

        elif self.obs_mode == 'sum_obs_limited':
            in_range = (dm < self.comm_radius) & (0 < dm)
            nr_neighbors = np.sum(in_range)

            local_obs = self.get_local_obs()
            local_obs[-1] = nr_neighbors / (self.n_agents - 1)

            sum_obs = np.zeros(self.dim_rec_o)

            sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
            sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
            sum_obs[0:nr_neighbors, 3] = (nh_size[in_range] - nr_neighbors) / (self.n_agents - 2) if self.n_agents > 2 \
                else np.zeros(nr_neighbors)
            sum_obs[0:nr_neighbors, 4] = np.cos(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 5] = np.sin(their_orientation[in_range])
            sum_obs[0:nr_neighbors, 6] = 1
            sum_obs[0:self.n_agents - 1, 7] = 1

            obs = np.hstack([sum_obs.flatten(), local_obs])
        elif self.obs_mode == 'eghn_acc':
            ind = np.where(dm == -1)[0][0]
            # local_obs = np.zeros(self.dim_local_o)
            # local_obs = self.get_local_obs()
            equ_obs = np.zeros(self.dim_equ_o)
            # add value normalization
            equ_obs[0] = self.state.p_pos[0] / self.world_size
            equ_obs[1] = self.state.p_pos[1] / self.world_size
            equ_obs[2] = self.state.p_vel[0] / self.max_lin_velocity
            equ_obs[3] = self.state.p_vel[1] / self.max_lin_velocity
            
            eghn_obs = np.zeros(self.dim_rec_o)
            eghn_obs[:, 0] = np.concatenate([dm[0:ind], dm[ind + 1:]]) / self.comm_radius
            eghn_obs[:, 1] = np.cos(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            eghn_obs[:, 2] = np.sin(np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]]))
            if self.local_mode:
                local_distances = np.concatenate([dm[0:ind], dm[ind + 1:]]) / self.comm_radius
                adjacency_vector = np.ones(self.n_agents) * -1  # 初始化为-1，表示没有连接
                adjusted_distances = np.insert(local_distances, ind, -1)  # 在当前智能体位置插入一个占位值
                for i, distance in enumerate(adjusted_distances):
                    if i != ind and distance <= self.obs_radius / self.comm_radius:
                        adjacency_vector[i] = 1  # 如果智能体间距离小于阈值，则设置为1
                obs = np.hstack([eghn_obs.flatten(), equ_obs.flatten(), adjacency_vector])
            else:
                obs = np.hstack([eghn_obs.flatten(), equ_obs.flatten()])
        elif self.obs_mode == 'hepn_local':
            ind = np.where(dm == -1)[0][0]
            
            # local_obs, invariant features.
            # torus为True时生效，当与墙面距离小于1时，计算与墙面的距离和角度的cos、sin；大于1时为[1, 0, 0]
            local_obs = np.zeros(self.dim_local_o)
            local_obs = self.get_local_obs()

            # equivariant features
            equ_obs = np.zeros(self.dim_equ_o)
            # add value normalization
            equ_obs[0] = self.state.p_pos[0] / self.world_size
            equ_obs[1] = self.state.p_pos[1] / self.world_size
            equ_obs[2] = self.state.p_vel[0] / self.max_lin_velocity
            equ_obs[3] = self.state.p_vel[1] / self.max_lin_velocity
            
            # invariant features (recevied information)
            dis_to_other = np.concatenate([dm[0:ind], dm[ind + 1:]])
            orientation = np.concatenate([my_orientation[0:ind], my_orientation[ind + 1:]])
            rec_obs = np.zeros(self.dim_rec_o)
            rec_obs[:, 0] = np.where(dis_to_other <= self.obs_radius, dis_to_other / self.obs_radius, 1)  # 通过obs_raadius归一化，超过的值置1
            rec_obs[:, 1] = np.where(dis_to_other <= self.obs_radius, np.cos(orientation), 0)  # 当距离超过obs_radius，置0
            rec_obs[:, 2] = np.where(dis_to_other <= self.obs_radius, np.sin(orientation), 0)  # 当距离超过obs_radius，置0

            if self.local_mode:
                # 局部观测模式，生成ind节点的邻接矩阵，与观测拼接
                # used for EGNN, EGHN and GNN-based models
                adjacency_vector = np.where(dm <= self.obs_radius, 1, 0)
                adjacency_vector[ind] = 0
                obs = np.hstack([equ_obs.flatten(), rec_obs.flatten(), local_obs.flatten(), adjacency_vector])
            else:
                obs = np.hstack([equ_obs.flatten(), rec_obs.flatten(), local_obs.flatten()])
        elif self.obs_mode == 'gnn_local':
            ind = np.where(dm == -1)[0][0]
            
            local_obs = np.zeros(self.dim_local_o)
            local_obs = self.get_local_obs()

            equ_obs = np.zeros(self.dim_equ_o)
            equ_obs[0] = self.state.p_pos[0] / self.world_size
            equ_obs[1] = self.state.p_pos[1] / self.world_size
            equ_obs[2] = self.state.p_vel[0] / self.max_lin_velocity
            equ_obs[3] = self.state.p_vel[1] / self.max_lin_velocity

            if self.local_mode:
                adjacency_vector = np.where(dm <= self.obs_radius, 1, -1)
                adjacency_vector[ind] = -1
                obs = np.hstack([local_obs.flatten(), equ_obs.flatten(), adjacency_vector])
            else:
                obs = np.hstack([local_obs.flatten(), equ_obs.flatten()])
        else:
            raise ValueError('histogram form must be 1D or 2D')
        return obs

    def get_local_obs_acc(self):
        local_obs = np.zeros(self.dim_local_o)
        local_obs[0] = self.state.p_vel[0] / self.max_lin_velocity
        local_obs[1] = self.state.p_vel[1] / self.max_lin_velocity
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
