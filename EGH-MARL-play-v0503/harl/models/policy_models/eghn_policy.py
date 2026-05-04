import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.model_util import *
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *
from harl.models.base.eghn import EGHN
from torch.distributions.normal import Normal
from harl.utils.envs_tools import get_shape_from_obs_space
import numpy as np


class EghnPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(EghnPolicy, self).__init__()
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        # env args
        act_dim = 2
        self.env_name = args["env_name"]
        obs_shape = get_shape_from_obs_space(obs_space)
        self.n_nodes = args["num_agents"]
        self.equ_nf = args.get("equ_nf", 4)
        self.inv_nf = int(obs_shape[0] - self.equ_nf)
        self.use_history = args["use_history"]
        self.windows_size = args["windows_size"]
        
        # experiments args
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch

        # args for network
        self.hidden_sizes = args["hidden_sizes"]
        self.n_cluster = args["n_cluster"]
        self.current_pooling_plan = None
        self.interaction_layer = args["interaction_layer"]
        self.pooling_layer = args["pooling_layer"]
        self.decoder_layer = args["decoder_layer"]
        self.flat = args["flat"]
        self.lp_loss = 0

        # args for local_obs_mode
        # self.comm_radius = args["comm_radius"]
        # self.world_size = args['world_size']
        # self.int_points_num = args.get("int_points_num", args.get("nr_evaders", 0))

        # 局部信息处理
        self.local_info_input = args["local_info_input"]
        self.local_info_output = args["local_info_output"]
        self.hpn_hidden = args["hpn_hidden"]
        self.local_mode = args["local_mode"]
        self.use_hpn = args["use_hpn"]
        self.local_tool = local_tool(self, args=args, net_type="egnn", id="actor")
        self.local_tool.dim_info_init(obs_shape[0])
        
        # network
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
        self.eghn_model = EGHN(in_node_nf=self.local_tool.inv_nf_new, 
                               in_edge_nf=1, hidden_nf=self.hidden_sizes[-1], device=device,
                               n_cluster=self.n_cluster, flat=self.flat, layer_per_block=self.interaction_layer,
                               layer_pooling=self.pooling_layer, activation=nn.SiLU(),
                               layer_decoder=self.decoder_layer, norm=True)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.to(device)

    def update_local_tool(self, obs_space):
        self.local_tool = local_tool(self, args=self.args, net_type="egnn", id="actor")
        obs_shape = get_shape_from_obs_space(obs_space)
        self.local_tool.dim_info_init(obs_shape[0])

    def update_edges(self):
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        # self.forward_edges, _ = self.get_edges_batch(self.n_threads)
        # self.eval_edges, _ = self.get_edges_batch(self.mini_batch_size)


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
        obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        obs = check(obs).to(**self.tpdv)
        # # 使用原始不变特征计算余弦相似度，构造边权
        # inv_fea = obs[:, self.local_tool.equ_nf:]
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        obs = self.local_tool.local_info_process(obs, self.local_module)

        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        
        rows, cols = self.local_tool.forward_edges
        edges = self.local_tool.forward_edges
        local_edge_index = self.local_tool.forward_edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        # local_edge_fea = edge_attr
        
        # norm_features = F.normalize(h, p=2, dim=1)
        # # 计算所有节点间的余弦相似度
        # cosine_similarity = torch.mm(norm_features, norm_features.t())
        # # 提取给定边对应的相似度
        # edge_similarities = cosine_similarity[rows, cols].unsqueeze(1).detach()
        # edge_attr = torch.clamp(edge_similarities, min=0.0)
        
        loc_pred, vel_pred, _ = self.eghn_model(loc, h, edges, edge_attr, local_edge_index, 
                                                edge_attr, n_node=self.n_nodes, v=vel, node_mask=None, 
                                                node_nums=None, flag=False)

        mu = vel_pred
        self.current_pooling_plan = self.eghn_model.current_pooling_plan
        self.h_edge_index = self.eghn_model.h_edge_index
        self.edge_weight = edge_attr
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        # action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        # policy = FixedNormal(mu, action_std)
        if deterministic:
            actions = policy.mode
        else:
            actions = policy.sample()
        # actions = (
        #     policy.mode()
        #     if deterministic
        #     else policy.sample()
        # )
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
        obs = check(obs).to(**self.tpdv)
        # # 使用原始不变特征计算余弦相似度，构造边权
        # inv_fea = obs[:, self.local_tool.equ_nf:]
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
        edges = self.local_tool.eval_edges
        local_edge_index = self.local_tool.eval_edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        local_edge_fea = edge_attr
        # norm_features = F.normalize(h, p=2, dim=1)
        # # 计算所有节点间的余弦相似度
        # cosine_similarity = torch.mm(norm_features, norm_features.t())
        # # 提取给定边对应的相似度
        # edge_similarities = cosine_similarity[rows, cols].unsqueeze(1).detach()
        # edge_attr = torch.clamp(edge_similarities, min=0.0)
        # local_edge_fea = edge_attr
        loc_pred, vel_pred, _ = self.eghn_model(loc, h, edges, edge_attr, local_edge_index, 
                                                local_edge_fea, n_node=self.n_nodes, v=vel, node_mask=None, 
                                                node_nums=None, flag=False)

        # self.lp_loss = self.eghn_model.structural_entropy
        # self.lp_loss = 0
        mu = vel_pred
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        # action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        # policy = FixedNormal(mu, action_std)
        action_distribution = policy
        dist_entropy = action_distribution.entropy().sum(axis=-1)
        dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        action_log_probs = policy.log_prob(action)

        return action_log_probs, dist_entropy, action_distribution
