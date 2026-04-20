import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space, get_shape_from_act_space
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from harl.models.base.model_util import *
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *
from harl.models.base.act import ACTLayer


class GATPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(GATPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.device = device
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.n_nodes = args["num_agents"]
        self.env_name = args["env_name"]
        obs_shape = get_shape_from_obs_space(obs_space)
        act_dim = get_shape_from_act_space(action_space)
        self.n_layers = args["n_layers"]
        self.head_num = args["head_num"]
        self.use_res = args["use_res"]
        self.use_eqc_flag = args["use_eqc_flag"]
        self.subgroup_num = int(args["subgroup_num"])
        self.rnn_mean_flag = args["rnn_mean_flag"]
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        
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
        self.local_tool = local_tool(self, args=args, net_type="gat", id="actor")
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
        
        self.embedding_in = nn.Linear(self.local_tool.inv_nf_new + self.local_tool.equ_nf, self.hidden_sizes[0])
        for i in range(0, self.n_layers):
            if i == 0:
                self.add_module("actor_gat_%d" % i, GATConv(int(self.hidden_sizes[0]), int(self.hidden_sizes[0]), heads=self.head_num))
            else:
                self.add_module("actor_gat_%d" % i, GATConv(int(self.hidden_sizes[0] * self.head_num), int(self.hidden_sizes[0]), heads=self.head_num))
        self.actor_fc1 = nn.Linear(self.hidden_sizes[0] * self.head_num, act_dim)
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[0] * self.head_num,
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(self.device)

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
        if self.use_eqc_flag:
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_obs_list = []
            for i in range(self.subgroup_num):
                angle = i * angle_increment
                if i == 0:
                    rotated_obs = obs.copy()
                else:
                    rotated_obs = rotation_obs3d(obs.copy(), angle, self.local_tool.equ_nf)
                rotated_obs_list.append(rotated_obs)
            obs = np.concatenate(rotated_obs_list, axis=0)
            
        obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, available_actions.shape[-1])
            available_actions = check(available_actions).to(**self.tpdv)
        
        obs = self.local_tool.local_info_process(obs, self.local_module)
        
        x = self.embedding_in(obs)
        for j in range(0, self.n_layers):
            x_res = x  # 保留原始输入
            x = self._modules["actor_gat_%d" % j](x, self.local_tool.forward_edges)
            if self.use_res:
                if not (self.head_num > 1 and j == 0):
                    x = x + x_res  # actor应用残差连接
            x = torch.tanh(x)
        
        if self.env_name == 'smacv2':
            actions, action_log_probs = self.act(
                x, available_actions, deterministic
            )
            actions = actions.reshape(self.n_threads, self.n_nodes, -1)
            action_log_probs = action_log_probs.reshape(self.n_threads, self.n_nodes, -1)
            return actions, action_log_probs, rnn_states

        mu = self.actor_fc1(x)
        if self.use_eqc_flag:
            parael_num = mu.shape[0]//self.subgroup_num
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_actions_list = []
            for i in range(self.subgroup_num):
                start_idx = parael_num * i
                end_idx = parael_num * (i + 1)
                if len(mu.shape) == 3:
                    actions = mu[start_idx:end_idx, :, :]
                else:
                    actions = mu[start_idx:end_idx, :]
                rotated_angle = 2 * np.pi - i * angle_increment
                if i == 0:
                    rotated_actions = actions
                else:
                    rotated_actions = rotation_action(actions, rotated_angle)
                rotated_actions_list.append(rotated_actions)
            mu = sum(rotated_actions_list) / self.subgroup_num
            
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        actions = policy.sample()
        action_log_probs = policy.log_prob(actions)
        actions = actions.reshape(self.n_threads, self.n_nodes, -1)
        action_log_probs = action_log_probs.reshape(self.n_threads, self.n_nodes, -1)
        return actions, action_log_probs, rnn_states

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
        obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        
        if self.use_eqc_flag:
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_obs_list = []
            for i in range(self.subgroup_num):
                angle = i * angle_increment
                if i == 0:
                    rotated_obs = obs.copy()
                else:
                    rotated_obs = rotation_obs2d(obs.copy(), angle, self.local_tool.equ_nf)
                rotated_obs_list.append(rotated_obs)
            obs = np.concatenate(rotated_obs_list, axis=0)
            
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        obs = self.local_tool.local_info_process(obs, self.local_module)

        x = self.embedding_in(obs)
        for j in range(0, self.n_layers):
            x_res = x  # 保留原始输入
            
            x = self._modules["actor_gat_%d" % j](x, self.local_tool.eval_edges)
            # 临时修改，用于评估
            # x = self._modules["actor_gat_%d" % j](x, self.local_tool.forward_edges)
            
            if self.use_res:
                if not (self.head_num > 1 and j == 0):
                    x = x + x_res  # actor应用残差连接
            x = torch.tanh(x)

        if self.env_name == 'smacv2':
            action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
                x,
                action,
                available_actions,
                active_masks=active_masks if self.use_policy_active_masks else None,
            )
            return action_log_probs, dist_entropy, action_distribution

        mu = self.actor_fc1(x)
        if self.use_eqc_flag:
            parael_num = mu.shape[0]//self.subgroup_num
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_actions_list = []
            for i in range(self.subgroup_num):
                start_idx = parael_num * i
                end_idx = parael_num * (i + 1)
                if len(mu.shape) == 3:
                    actions = mu[start_idx:end_idx, :, :]
                else:
                    actions = mu[start_idx:end_idx, :]
                rotated_angle = 2 * np.pi - i * angle_increment
                if i == 0:
                    rotated_actions = actions
                else:
                    rotated_actions = rotation_action(actions, rotated_angle)
                rotated_actions_list.append(rotated_actions)
            mu = sum(rotated_actions_list) / self.subgroup_num
            
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        
        action_distribution = policy
        dist_entropy = action_distribution.entropy().sum(axis=-1)
        dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        action_log_probs = policy.log_prob(action)
        return action_log_probs, dist_entropy, action_distribution
    
