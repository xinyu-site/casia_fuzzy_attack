import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space
from torch.distributions.normal import Normal
from harl.models.base.model_util import *
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *
from harl.models.base.egnn import EGNN
from harl.models.base.mean_embedding import MyMLP

class EgnnV2Policy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(EgnnV2Policy, self).__init__()
        self.env_name = args["env_name"]
        self.hidden_nf = args["hidden_sizes"]
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        
        # EQC baseline
        self.use_eqc_flag = args["use_eqc_flag"]
        self.subgroup_num = int(args["subgroup_num"])
        self.rnn_mean_flag = args["rnn_mean_flag"]
        
        self.n_nodes = args["num_agents"]
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        
        # env args
        act_dim = 2
        obs_shape = get_shape_from_obs_space(obs_space)
        self.flat = args["flat"]
        self.in_edge_nf = args["in_edge_nf"]
        self.n_layers = args["n_layers"]

        # 时序信息处理
        self.use_history = args["use_history"]
        self.windows_size = args["windows_size"]
        self.c = 1

        # 局部信息处理
        self.local_info_input = args["local_info_input"]
        self.local_info_output = args["local_info_output"]
        self.hpn_hidden = args["hpn_hidden"]
        self.local_mode = args["local_mode"]
        self.use_hpn = args["use_hpn"]
        self.local_tool = local_tool(self, args=args, net_type="egnn", id="actor")
        self.local_tool.dim_info_init(obs_shape[0])
        if self.local_mode:
            self.local_module = nn.ModuleList()
            for i in range(len(self.local_info_output)):
                if self.use_hpn:
                    layer = HyperMLP(input_dim=self.local_info_input[i], output_dim=self.local_info_output[i], hyper_hidden_dim=self.hpn_hidden[i])
                else:
                    layer = LocalMLP(input_dim=self.local_info_input[i], output_dim=self.local_info_output[i])
                self.local_module.append(layer)
        else:
            self.local_module = None
            
        self.egnn_model = EGNN(n_layers=self.n_layers, in_edge_nf=self.in_edge_nf, in_node_nf=self.local_tool.inv_nf_new, hidden_nf=self.hidden_nf[0], device=device,
                                flat=self.flat,  with_v=True, activation=nn.SiLU(), norm=True)
        if self.use_history:
            # 仅等变数据采用时序矩阵计算Attention
            self.history_weight = AttentionLayer(self.windows_size, self.local_tool.equ_nf, self.local_tool.inv_nf_old+self.local_tool.equ_nf)
            # 仅等变数据采用时序矩阵加权
            # self.history_weight = PartLinearLayer(self.windows_size, self.equ_nf, self.inv_nf_old+self.equ_nf)
            # 所有数据均采用时序加权
            # self.history_weight = AllLinearLayer(self.windows_size, self.inv_nf_old+self.equ_nf)
            # 仅使用最后一个时刻的数据，检查数据正确性
            # self.history_weight = StaticWeightLinearLayer(input_features=self.windows_size, output_features=self.c)
            
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.to(device)

    def update_local_tool(self, obs_space):
        self.local_tool = local_tool(self, args=self.args, net_type="egnn", id="actor")
        obs_shape = get_shape_from_obs_space(obs_space)
        self.local_tool.dim_info_init(obs_shape[0])

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        
        obs = self.local_tool.trans_info2local_actor(obs)
        if self.use_history:
            obs = obs.reshape(-1, (self.local_tool.inv_nf_old + self.local_tool.equ_nf) * self.windows_size)
            obs = check(obs).to(**self.tpdv)
            obs = obs.reshape(-1, self.windows_size, (self.local_tool.inv_nf_old + self.local_tool.equ_nf))
            obs = torch.transpose(obs, 1, 2)
            obs = self.history_weight(obs)
        else:
            obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        obs = self.local_tool.local_info_process(obs, self.local_module)
        
        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        rows, cols = self.local_tool.forward_edges
        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        loc_pred, vel_pred, _ = self.egnn_model(loc, h, self.local_tool.forward_edges, edge_attr, v=vel)
        mu = vel_pred
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        actions = policy.sample()
        action_log_probs = policy.log_prob(actions)
        actions = actions.reshape(self.n_threads, self.n_nodes, -1)
        action_log_probs = action_log_probs.reshape(self.n_threads, self.n_nodes, -1)
        return actions, action_log_probs, rnn_states
    
    def forward_decided(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        #print(obs.shape) 10*10*44 
        obs = self.local_tool.trans_info2local_actor(obs)
        #print(obs.shape) 10*10*34
        #print(obs.shape)
        if self.use_history:
            obs = obs.reshape(-1, (self.local_tool.inv_nf_old + self.local_tool.equ_nf) * self.windows_size)
            obs = check(obs).to(**self.tpdv)
            obs = obs.reshape(-1, self.windows_size, (self.local_tool.inv_nf_old + self.local_tool.equ_nf))
            obs = torch.transpose(obs, 1, 2)
            obs = self.history_weight(obs)
        else:
            obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        #print(obs.shape)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        # print(obs.shape) 10*34
        obs = self.local_tool.local_info_process(obs, self.local_module)
        #print(obs.shape)
        
        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        #print(h.shape) 100*19
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        rows, cols = self.local_tool.forward_edges
        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        loc_pred, vel_pred, _ = self.egnn_model(loc, h, self.local_tool.forward_edges, edge_attr, v=vel)

        return vel_pred, rnn_states
    
    def forward_grd(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        
        obs = self.local_tool.trans_info2local_actor(obs)
        if self.use_history:
            obs = obs.reshape(-1, (self.local_tool.inv_nf_old + self.local_tool.equ_nf) * self.windows_size)
            obs = check(obs).to(**self.tpdv)
            obs = obs.reshape(-1, self.windows_size, (self.local_tool.inv_nf_old + self.local_tool.equ_nf))
            obs = torch.transpose(obs, 1, 2)
            obs = self.history_weight(obs)
        else:
            obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        obs = self.local_tool.local_info_process(obs, self.local_module)
        
        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        rows, cols = self.local_tool.forward_edges
        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
        loc_pred, vel_pred, _ = self.egnn_model(loc, h, self.local_tool.forward_edges, edge_attr, v=vel)
        #mu = vel_pred
        #std = torch.exp(self.log_std)
        #policy = Normal(mu, std)
        actions = vel_pred
        #action_log_probs = policy.log_prob(actions)
        actions = actions.reshape(self.n_threads, self.n_nodes, -1)
        #action_log_probs = action_log_probs.reshape(self.n_threads, self.n_nodes, -1)
        return actions,  rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = self.local_tool.trans_info2local_actor(obs, evaluate_mode=True)
        if self.use_history:
            obs = obs.reshape(-1, (self.local_tool.inv_nf_old + self.local_tool.equ_nf) * self.windows_size)
            obs = check(obs).to(**self.tpdv)
            obs = obs.reshape(-1, self.windows_size, (self.local_tool.inv_nf_old + self.local_tool.equ_nf))
            obs = torch.transpose(obs, 1, 2)
            obs = self.history_weight(obs)
        else:
            obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
            
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        obs = self.local_tool.local_info_process(obs, self.local_module)
        
        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        rows, cols = self.local_tool.eval_edges
        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        loc_pred, vel_pred, _ = self.egnn_model(loc, h, self.local_tool.eval_edges, edge_attr, v=vel)
        mu = vel_pred
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        action_distribution = policy
        dist_entropy = action_distribution.entropy().sum(axis=-1)
        dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        action_log_probs = policy.log_prob(action)
        return action_log_probs, dist_entropy, action_distribution
    
