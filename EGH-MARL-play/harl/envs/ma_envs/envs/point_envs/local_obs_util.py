import torch
import torch.nn as nn
import numpy as np
from harl.models.base.model_util import *
import networkx as nx
    
class local_tool():
    def __init__(self, agent, args, net_type="egnn", id="actor"):
        self.args = args
        self.inv_nf_old = 0
        self.inv_nf_new = 0
        self.id = id # "actor", "critic"
        self.n_nodes = agent.n_nodes
        self.env_name = agent.env_name
        env_list = ["rendezvous", "navigation", "navigation_v2", "pursuit", "cover", "mujoco3d", "smacv2"]
        assert self.env_name in env_list
        
        self.mini_batch_size = agent.mini_batch_size
        self.n_threads = agent.n_threads
        self.use_history = agent.use_history
        self.windows_size = agent.windows_size

        self.int_points_num = args["int_points_num"]
        self.comm_radius = args["comm_radius"]
        self.obs_radius = args["obs_radius"]
        self.equ_nf = args["equ_nf"]
        self.local_mode = args["local_mode"]
        self.critic_use_local_module = args["critic_use_local_module"]
        self.local_info_input = args["local_info_input"]
        self.local_info_output = args["local_info_output"]
        if self.id == "actor":
            self.edge_tool = edge_tool(agent, net_type)
            self.edge_tool.actor_num_mini_batch = agent.actor_num_mini_batch
            self.forward_edges = self.edge_tool.get_edges_batch(self.n_threads)
            self.eval_edges = self.edge_tool.get_edges_batch(self.mini_batch_size)
        else:
            self.edge_tool = edge_tool(agent, net_type)
            self.collect_edges = self.edge_tool.get_edges_batch(self.n_threads)
            self.train_edges = self.edge_tool.get_edges_batch(self.mini_batch_size)
            
        info2local_mapping = {
            "rendezvous": self.rendezvous_info2local,
            "navigation": self.navigation_info2local,
            "navigation_v2": self.navigation_v2_info2local,
            "pursuit": self.pursuit_info2local,
            "cover": self.cover_info2local,
            "mujoco3d": self.mujoco3d_info2local
        }
        self.info2local_func = info2local_mapping.get(self.env_name)
        
        localinfo_process_mapping = {
            "rendezvous": self.rendezvous_localinfo_process,
            "navigation": self.navigation_localinfo_process,
            "navigation_v2": self.navigation_v2_localinfo_process,
            "pursuit": self.pursuit_localinfo_process,
            "cover": self.cover_localinfo_process,
            "mujoco3d": self.mujoco3d_localinfo_process
        }
        self.localinfo_process_func = localinfo_process_mapping.get(self.env_name)
        
    def update_edges(self):
        self.n_threads = 1
        self.mini_batch_size = self.n_threads * self.edge_tool.episode_length // self.edge_tool.actor_num_mini_batch
        self.forward_edges = self.edge_tool.get_edges_batch(self.n_threads)
        self.eval_edges = self.edge_tool.get_edges_batch(self.mini_batch_size)
        
    def dim_info_init(self, obs_dim):
        if self.local_mode:
            self.inv_nf_old = obs_dim - self.equ_nf - self.n_nodes
            if self.critic_use_local_module or self.id == "actor":
                if self.env_name == "mujoco3d":
                    self.inv_nf_new = self.inv_nf_old
                else:
                    self.inv_nf_new = sum(self.local_info_output) + 3 * (not self.args["torus"])
                # if self.env_name == "cover":
                #     self.inv_nf_new = self.inv_nf_new + 3
            else:
                self.inv_nf_new = self.inv_nf_old
        else:
            self.inv_nf_old = obs_dim - self.equ_nf
            self.inv_nf_new = self.inv_nf_old
            
    def trans_info2local_critic(self, cent_obs):
        if not self.local_mode:
            return cent_obs
        else:
            if self.use_history:
                cent_obs = np.reshape(cent_obs, (-1, self.windows_size, self.n_nodes, (self.equ_nf + self.inv_nf_old + self.n_nodes)))
                cent_obs = np.transpose(cent_obs, (0, 2, 3, 1))
                cent_obs = cent_obs.reshape(-1, (self.inv_nf_old + self.equ_nf + self.n_nodes), self.windows_size)
                cent_obs = cent_obs[:, :-self.n_nodes, :]
                return np.reshape(cent_obs, (-1, self.windows_size*self.n_nodes*(self.equ_nf + self.inv_nf_old)))
            else:
                cent_obs = np.reshape(cent_obs, (-1, self.n_nodes, self.equ_nf + self.inv_nf_old + self.n_nodes))
                cent_obs = cent_obs[:, :, :-self.n_nodes]
                if self.env_name == "mujoco3d":
                    self.train_edges = self.edge_tool.get_edges_batch(self.mini_batch_size)
                    self.collect_edges = self.edge_tool.get_edges_batch(self.n_threads)
                return np.reshape(cent_obs, (-1, self.n_nodes*(self.equ_nf + self.inv_nf_old)))
    
    def trans_info2local_actor(self, obs, evaluate_mode=False):
        if not self.local_mode:
            return obs
        else:
            # 检查输入类型，选择合适的操作
            is_tensor = torch.is_tensor(obs)
            
            if self.use_history:
                if is_tensor:
                    obs = obs.reshape(-1, (self.inv_nf_old + self.equ_nf + self.n_nodes) * self.windows_size)
                    obs = obs.reshape(-1, self.windows_size, (self.inv_nf_old + self.equ_nf + self.n_nodes))
                    obs = torch.transpose(obs, 1, 2)
                else:
                    obs = obs.reshape(-1, (self.inv_nf_old + self.equ_nf + self.n_nodes) * self.windows_size)
                    obs = obs.reshape(-1, self.windows_size, (self.inv_nf_old + self.equ_nf + self.n_nodes))
                    obs = np.transpose(obs, (0, 2, 1))
                
                obs_list = []
                edge_list = []

                for i in range(self.windows_size):
                    temp_obs = obs[:, :, i]
                    if is_tensor:
                        temp_obs = temp_obs.reshape(-1, self.n_nodes, (self.equ_nf + self.inv_nf_old + self.n_nodes))
                    else:
                        temp_obs = np.reshape(temp_obs, (-1, self.n_nodes, (self.equ_nf + self.inv_nf_old + self.n_nodes)))
                    
                    if evaluate_mode:
                        if is_tensor:
                            local_obs = self.info2local_func(temp_obs.reshape((self.mini_batch_size, self.n_nodes, -1)))
                        else:
                            local_obs = self.info2local_func(temp_obs.reshape((self.mini_batch_size, self.n_nodes, -1)))
                        eval_edges = self.edge_tool.get_edges_batch(self.mini_batch_size, local_obs)
                        edge_list.append(eval_edges)

                    else:
                        local_obs = self.info2local_func(temp_obs)
                        forward_edges = self.edge_tool.get_edges_batch(self.n_threads, local_obs)
                        edge_list.append(forward_edges)

                    obs_list.append(local_obs[:, :, :-self.n_nodes])
                
                if evaluate_mode:   
                    self.eval_edges = edge_list[-1]
                else:
                    self.forward_edges = edge_list[-1]
                
                if is_tensor:
                    local_obs = torch.cat(obs_list, dim=2)
                else:
                    local_obs = np.concatenate(obs_list, axis=2)
                return local_obs
            else:
                if evaluate_mode:
                    if is_tensor:
                        local_obs = self.info2local_func(obs.reshape((self.mini_batch_size, self.n_nodes, -1)))
                    else:
                        local_obs = self.info2local_func(obs.reshape((self.mini_batch_size, self.n_nodes, -1)))
                    self.eval_edges = self.edge_tool.get_edges_batch(self.mini_batch_size, local_obs)
                else:
                    local_obs = self.info2local_func(obs)
                    self.forward_edges = self.edge_tool.get_edges_batch(self.n_threads, local_obs)
                return local_obs[:, :, :-self.n_nodes]

    def local_info_process(self, obs, local_module):
        if not self.local_mode:
            return obs
        elif (not self.critic_use_local_module) and (self.id == "critic"):
            return obs
        else:
            local_obs = self.localinfo_process_func(obs, local_module)
            return local_obs
    
    # envName_local2local:将全局信息的观测转化为局部信息的函数，通过将观测范围之外的信息置0实现
    # def rendezvous_info2local(self, obs):
    #     t_obs = obs.copy()
    #     batch_size, n_agents, _ = t_obs.shape
    #     distance_threshold = self.obs_radius / self.comm_radius
    #     distance = t_obs[:, :, :(n_agents - 1) * 3].reshape(batch_size, n_agents, n_agents - 1, 3)[:, :, :, 0]
    #     temp1 = t_obs[:, :, :(n_agents - 1) * 3].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
    #     temp1[distance > distance_threshold, :] = 0
    #     temp1 = temp1.reshape(batch_size, n_agents, -1)
    #     t_obs[:, :, :(n_agents - 1) * 3] = temp1
    #     return t_obs
    
    # 适用于obs_mode为hepn_local，local_mode为True
    def rendezvous_info2local(self, obs):
        # 检查输入类型
        is_tensor = torch.is_tensor(obs)
        
        if is_tensor:
            t_obs = obs.clone()  # tensor使用clone保持梯度
            batch_size, n_agents, _ = t_obs.shape
            temp = t_obs[
                :, :, self.equ_nf:(n_agents - 1) * 3 + self.equ_nf
            ].clone().reshape(batch_size, n_agents, n_agents - 1, 3)  # 智能体接收到的其余智能体信息
            # 使用torch.where替代np.where
            mask = torch.ones_like(temp) * torch.tensor([1., 0., 0.]).to(temp.device)
            temp = torch.where(temp == mask, torch.zeros_like(temp), temp)
            temp = temp.reshape(batch_size, n_agents, -1)
            t_obs[:, :, self.equ_nf:(n_agents - 1) * 3 + self.equ_nf] = temp
            return t_obs
        else:
            # 原有的numpy逻辑
            t_obs = obs.copy()
            batch_size, n_agents, _ = t_obs.shape
            temp = t_obs[
                :, :, self.equ_nf:(n_agents - 1) * 3 + self.equ_nf
            ].copy().reshape(batch_size, n_agents, n_agents - 1, 3)  # 智能体接收到的其余智能体信息
            temp = np.where(temp == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp)
            temp = temp.reshape(batch_size, n_agents, -1)
            t_obs[:, :, self.equ_nf:(n_agents - 1) * 3 + self.equ_nf] = temp
            return t_obs

    # def navigation_info2local(self, obs):
    #     t_obs = obs.copy()
    #     batch_size, n_agents, _ = t_obs.shape
    #     distance_threshold = self.obs_radius / self.comm_radius
    #     distance = t_obs[:, :, :(n_agents - 1) * 3].reshape(batch_size, n_agents, n_agents - 1, 3)[:, :, :, 0]
    #     int_distance = t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3].reshape(batch_size, n_agents, 1, 3)[:, :, :, 0]
    #     temp1 = t_obs[:, :, :(n_agents - 1) * 3].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
    #     temp1[distance > distance_threshold, :] = 0
    #     temp1 = temp1.reshape(batch_size, n_agents, -1)
        
    #     temp2 = t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3].copy().reshape(batch_size, n_agents, 1, 3)
    #     temp2[int_distance > distance_threshold, :] = 0
    #     temp2 = temp2.reshape(batch_size, n_agents, -1)
    #     t_obs[:, :, :(n_agents - 1) * 3] = temp1
    #     t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3] = temp2
    #     return t_obs

    # 适用于obs_mode为hepn_local，local_mode为True
    def navigation_info2local(self, obs):
        # 检查输入类型
        is_tensor = torch.is_tensor(obs)
        
        if is_tensor:
            t_obs = obs.clone()  # tensor使用clone保持梯度
            batch_size, n_agents, _ = t_obs.shape
            temp1 = t_obs[
                :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
            ].clone().reshape(batch_size, n_agents, n_agents - 1, 3)
            # 使用torch.where替代np.where
            mask = torch.ones_like(temp1) * torch.tensor([1., 0., 0.]).to(temp1.device)
            temp1 = torch.where(temp1 == mask, torch.zeros_like(temp1), temp1)
            temp1 = temp1.reshape(batch_size, n_agents, -1)
            
            temp2 = t_obs[
                :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3
            ].clone().reshape(batch_size, n_agents, 1, 3)
            temp2 = torch.where(temp2 == mask, torch.zeros_like(temp2), temp2)
            temp2 = temp2.reshape(batch_size, n_agents, -1)

            t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
            t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3] = temp2
            return t_obs
        else:
            # 原有的numpy逻辑
            t_obs = obs.copy()
            batch_size, n_agents, _ = t_obs.shape
            temp1 = t_obs[
                :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
            ].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
            temp1 = np.where(temp1 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp1)
            temp1 = temp1.reshape(batch_size, n_agents, -1)
            
            temp2 = t_obs[
                :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3
            ].copy().reshape(batch_size, n_agents, 1, 3)
            temp2 = np.where(temp2 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp2)
            temp2 = temp2.reshape(batch_size, n_agents, -1)

            t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
            t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3] = temp2
            return t_obs
    
    # def navigation_v2_info2local(self, obs):
    #     t_obs = obs.copy()
    #     batch_size, n_agents, obs_dim = t_obs.shape
    #     obstacles_num = (obs_dim - n_agents - (n_agents-1) * 3 - 4 - 3)//3
    #     distance_threshold = self.obs_radius / self.comm_radius
    #     distance = t_obs[:, :, :(n_agents - 1) * 3].reshape(batch_size, n_agents, n_agents - 1, 3)[:, :, :, 0]
    #     int_distance = t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3].reshape(batch_size, n_agents, 1, 3)[:, :, :, 0]
    #     obstacles_distance = t_obs[:, :, (n_agents-1)*3+3:(n_agents-1+obstacles_num)*3+3].reshape(batch_size, n_agents, obstacles_num, 3)[:, :, :, 0]
    #     temp1 = t_obs[:, :, :(n_agents - 1) * 3].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
    #     temp1[distance > distance_threshold, :] = 0
    #     temp1 = temp1.reshape(batch_size, n_agents, -1)
        
    #     temp2 = t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3].copy().reshape(batch_size, n_agents, 1, 3)
    #     temp2[int_distance > distance_threshold, :] = 0
    #     temp2 = temp2.reshape(batch_size, n_agents, -1)
        
    #     temp3 = t_obs[:, :, (n_agents-1)*3+3:(n_agents-1+obstacles_num)*3+3].copy().reshape(batch_size, n_agents, obstacles_num, 3)
    #     temp3[obstacles_distance > distance_threshold, :] = 0
    #     temp3 = temp3.reshape(batch_size, n_agents, -1)
        
    #     t_obs[:, :, :(n_agents - 1) * 3] = temp1
    #     t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3] = temp2
    #     t_obs[:, :, (n_agents-1)*3+3:(n_agents-1+obstacles_num)*3+3] = temp3
    #     return t_obs

    # 适用于obs_mode为hepn_local，local_mode为True
    def navigation_v2_info2local(self, obs):
        t_obs = obs.copy()
        batch_size, n_agents, obs_dim = t_obs.shape
        obstacles_num = (obs_dim - n_agents - (n_agents - 1) * 3 - 4 - 3 - 3 * int(not self.args["torus"])) // 3

        temp1 = t_obs[
            :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
        ].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
        temp1 = np.where(temp1 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp1)
        temp1 = temp1.reshape(batch_size, n_agents, -1)
        
        temp2 = t_obs[
            :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3
        ].copy().reshape(batch_size, n_agents, 1, 3)
        temp2 = np.where(temp2 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp2)
        temp2 = temp2.reshape(batch_size, n_agents, -1)
        
        temp3 = t_obs[
            :, :, self.equ_nf + (n_agents - 1) * 3 + 3 : self.equ_nf + (n_agents - 1 + obstacles_num) * 3 + 3
        ].copy().reshape(batch_size, n_agents, obstacles_num, 3)
        temp3 = np.where(temp3 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp3)
        temp3 = temp3.reshape(batch_size, n_agents, -1)
        
        t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
        t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3] = temp2
        t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 + 3 : self.equ_nf + (n_agents - 1 + obstacles_num) * 3 + 3] = temp3
        return t_obs

    # def pursuit_info2local(self, obs):
    #     t_obs = obs.copy()
    #     batch_size, n_agents, _ = t_obs.shape
    #     distance_threshold = self.obs_radius / self.comm_radius
    #     distance = t_obs[:, :, :(n_agents - 1) * 3].reshape(batch_size, n_agents, n_agents - 1, 3)[:, :, :, 0]
    #     int_distance = t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3].reshape(batch_size, n_agents, 1, 3)[:, :, :, 0]
    #     temp1 = t_obs[:, :, :(n_agents - 1) * 3].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
    #     temp1[distance > distance_threshold, :] = 0
    #     temp1 = temp1.reshape(batch_size, n_agents, -1)
        
    #     temp2 = t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3].copy().reshape(batch_size, n_agents, 1, 3)
    #     temp2[int_distance > distance_threshold, :] = 0
    #     temp2 = temp2.reshape(batch_size, n_agents, -1)
    #     t_obs[:, :, :(n_agents - 1) * 3] = temp1
    #     t_obs[:, :, (n_agents-1)*3:(n_agents-1)*3+3] = temp2
    #     return t_obs

    # 适用于obs_mode为hepn_local，local_mode为True
    def pursuit_info2local(self, obs):
        # 检查输入类型
        is_tensor = torch.is_tensor(obs)
        
        if is_tensor:
            t_obs = obs.clone()  # tensor使用clone保持梯度
            batch_size, n_agents, _ = t_obs.shape
            
            temp1 = t_obs[
                :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
            ].clone().reshape(batch_size, n_agents, n_agents - 1, 3)
            # 使用torch.where替代np.where
            mask = torch.ones_like(temp1) * torch.tensor([1., 0., 0.]).to(temp1.device)
            temp1 = torch.where(temp1 == mask, torch.zeros_like(temp1), temp1)
            temp1 = temp1.reshape(batch_size, n_agents, -1)
            
            temp2 = t_obs[
                :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3
            ].clone().reshape(batch_size, n_agents, 1, 3)
            temp2 = torch.where(temp2 == mask, torch.zeros_like(temp2), temp2)
            temp2 = temp2.reshape(batch_size, n_agents, -1)
            
            t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
            t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3] = temp2
            return t_obs
        else:
            # 原有的numpy逻辑
            t_obs = obs.copy()
            batch_size, n_agents, _ = t_obs.shape
            
            temp1 = t_obs[
                :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
            ].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
            temp1 = np.where(temp1 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp1)
            temp1 = temp1.reshape(batch_size, n_agents, -1)
            
            temp2 = t_obs[
                :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3
            ].copy().reshape(batch_size, n_agents, 1, 3)
            temp2 = np.where(temp2 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp2)
            temp2 = temp2.reshape(batch_size, n_agents, -1)
            
            t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
            t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents - 1) * 3 + 3] = temp2
            return t_obs

    # def cover_info2local(self, obs):
    #     t_obs = obs.copy()
    #     batch_size, n_agents, _ = t_obs.shape
    #     distance_threshold = self.obs_radius / self.comm_radius
    #     distance = t_obs[:, :, :(n_agents - 1) * 3].reshape(batch_size, n_agents, n_agents - 1, 3)[:, :, :, 0]
    #     int_point_distances = t_obs[:, :, (n_agents-1)*3:(n_agents+ self.int_points_num-1)*3].reshape(batch_size, n_agents, self.int_points_num, 3)[:, :, :, 0]

    #     temp1 = t_obs[:, :, :(n_agents - 1) * 3].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
    #     temp1[distance > distance_threshold, :] = 0
    #     temp1 = temp1.reshape(batch_size, n_agents, -1)
        
    #     temp2 = t_obs[:, :, (n_agents-1)*3:(n_agents+self.int_points_num-1)*3].copy().reshape(batch_size, n_agents, 30, 3)
    #     temp2[int_point_distances > distance_threshold, :] = 0
    #     temp2 = temp2.reshape(batch_size, n_agents, -1)
        
    #     t_obs[:, :, :(n_agents - 1) * 3] = temp1
    #     t_obs[:, :, (n_agents-1)*3:(n_agents+self.int_points_num-1)*3] = temp2
    #     return t_obs

    # 适用于obs_mode为hepn_local，local_mode为True
    def cover_info2local(self, obs):
        # 检查输入类型
        is_tensor = torch.is_tensor(obs)
        
        if is_tensor:
            t_obs = obs.clone()  # tensor使用clone保持梯度
            batch_size, n_agents, _ = t_obs.shape

            temp1 = t_obs[
                :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
            ].clone().reshape(batch_size, n_agents, n_agents - 1, 3)
            # 使用torch.where替代np.where
            mask = torch.ones_like(temp1) * torch.tensor([1., 0., 0.]).to(temp1.device)
            temp1 = torch.where(temp1 == mask, torch.zeros_like(temp1), temp1)
            temp1 = temp1.reshape(batch_size, n_agents, -1)
            
            temp2 = t_obs[
                :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents + self.int_points_num - 1) * 3
            ].clone().reshape(batch_size, n_agents, 30, 3)
            temp2 = torch.where(temp2 == mask, torch.zeros_like(temp2), temp2)
            temp2 = temp2.reshape(batch_size, n_agents, -1)
            
            t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
            t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents + self.int_points_num - 1) * 3] = temp2
            return t_obs
        else:
            # 原有的numpy逻辑
            t_obs = obs.copy()
            batch_size, n_agents, _ = t_obs.shape

            temp1 = t_obs[
                :, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3
            ].copy().reshape(batch_size, n_agents, n_agents - 1, 3)
            temp1 = np.where(temp1 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp1)
            temp1 = temp1.reshape(batch_size, n_agents, -1)
            
            temp2 = t_obs[
                :, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents + self.int_points_num - 1) * 3
            ].copy().reshape(batch_size, n_agents, 30, 3)
            temp2 = np.where(temp2 == np.array([1., 0., 0.]), np.array([0., 0., 0.]), temp2)
            temp2 = temp2.reshape(batch_size, n_agents, -1)
            
            t_obs[:, :, self.equ_nf:self.equ_nf + (n_agents - 1) * 3] = temp1
            t_obs[:, :, self.equ_nf + (n_agents - 1) * 3 : self.equ_nf + (n_agents + self.int_points_num - 1) * 3] = temp2
            return t_obs

    def mujoco3d_info2local(self, obs):
        return obs

    # envName_localinfo_process:处理局部信息的函数，目前采用矩阵求和处理的方式实现
    def rendezvous_localinfo_process(self, local_obs, local_module):
        part1 = local_obs[:, :self.equ_nf]  # 等变特征

        part2 = local_obs[
            :, self.equ_nf:self.equ_nf + (self.n_nodes - 1) * 3
        ].reshape(-1, self.n_nodes - 1, 3)  # 邻居智能体特征
        output2 = local_module[0](part2)

        part3 = local_obs[:, self.equ_nf + (self.n_nodes - 1) * 3:]  # 局部观测
        
        local_obs = torch.cat((part1, output2, part3), dim=1)

        return local_obs

    def navigation_localinfo_process(self, local_obs, local_module):
        part1 = local_obs[:, :self.equ_nf]  # 等变特征
        
        part2 = local_obs[
            :, self.equ_nf:self.equ_nf + (self.n_nodes - 1) * 3
        ].reshape(-1, self.n_nodes - 1, 3)  # 邻居智能体特征
        output2 = local_module[0](part2)

        part3 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1) * 3 : self.equ_nf + (self.n_nodes - 1) * 3 + 3
        ].reshape(-1, 1, 3)  #  目标点特征
        output3 = local_module[1](part3)
        
        part4 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1) * 3 + 3 :
        ]

        local_obs = torch.cat((part1, output2, output3, part4), dim=1)
        return local_obs
    
    def navigation_v2_localinfo_process(self, local_obs, local_module):
        part1 = local_obs[:, :self.equ_nf]  # 等变特征

        obstacles_num = (local_obs.shape[1] - (self.n_nodes - 1) * 3 - 4 - 3 - 3 * int(not self.args["torus"])) // 3  # 障碍物个数
        part2 = local_obs[
            :, self.equ_nf:self.equ_nf + (self.n_nodes - 1) * 3
        ].reshape(-1, self.n_nodes - 1, 3)  # 邻居智能体特征
        output2 = local_module[0](part2)

        part3 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1) * 3 : self.equ_nf + (self.n_nodes - 1) * 3 + 3
        ].reshape(-1, 1, 3)  # 目标点特征
        output3 = local_module[1](part3)

        part4 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1) * 3 + 3 : self.equ_nf + (self.n_nodes - 1 + obstacles_num) * 3 + 3
        ].reshape(-1, obstacles_num, 3)  # 障碍物特征
        output4 = local_module[2](part4)

        part5 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1 + obstacles_num) * 3 + 3:
        ] # 局部观测

        local_obs = torch.cat((part1, output2, output3, output4, part5), dim=1)
        
        return local_obs

    def pursuit_localinfo_process(self, local_obs, local_module):
        part1 = local_obs[:, :self.equ_nf]  # 等变特征

        part2 = local_obs[
            :, self.equ_nf:self.equ_nf + (self.n_nodes - 1) * 3
        ].reshape(-1, self.n_nodes - 1, 3) # 邻居智能体特征
        output2 = local_module[0](part2)

        part3 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1) * 3 : self.equ_nf + (self.n_nodes - 1) * 3 + 3
        ].reshape(-1, 1, 3)  # 逃跑者特征
        output3 = local_module[1](part3)

        part4 = local_obs[:, self.equ_nf + (self.n_nodes - 1) * 3 + 3:]  # 局部观测

        local_obs = torch.cat((part1, output2, output3, part4), dim=1)
        
        return local_obs

    def cover_localinfo_process(self, local_obs, local_module):
        part1 = local_obs[:, :self.equ_nf]  # 等变特征

        part2 = local_obs[
            :, self.equ_nf:self.equ_nf + (self.n_nodes - 1) * 3
        ].reshape(-1, self.n_nodes - 1, 3) # 邻居智能体特征
        output2 = local_module[0](part2)

        part3 = local_obs[
            :, self.equ_nf + (self.n_nodes - 1) * 3 : self.equ_nf + (self.n_nodes + self.int_points_num - 1) * 3
        ].reshape(-1, self.int_points_num, 3) # 兴趣点特征
        output3 = local_module[1](part3)
        
        part4 = local_obs[
            :, self.equ_nf + (self.n_nodes + self.int_points_num - 1) * 3
        ]
        
        local_obs = torch.cat((part1, output2, output3, part4), dim=1)
        
        return local_obs
    
    def mujoco3d_localinfo_process(self, local_obs, local_module):
        return local_obs