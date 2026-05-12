import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space
from harl.models.base.model_util import *
from harl.models.base.hierarchy import HMF
from harl.models.base.act import ACTLayer
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *


class HmfPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(HmfPolicy, self).__init__()
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        # env args
        act_dim = get_shape_from_act_space(action_space)
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
        self.n_layers = args["n_layers"]
        attn_heads = args["attn_heads"]
        attn_dropout = args["attn_dropout"]
        ffn_dropout = args["ffn_dropout"]
        attn_ffn_hidden_scale = args["attn_ffn_hidden_scale"]

        self.flat = args["flat"]
        self.use_res = args["use_res"]
        # self.lp_loss = 0
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.initialization_method = args["initialization_method"]
        self.gain = args["gain"]

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

        self.hie_model = HMF(
            in_node_nf=self.local_tool.inv_nf_new + self.local_tool.equ_nf,
            hidden_nf=self.hidden_sizes[-1],
            n_cluster=self.n_cluster,
            n_layer=self.n_layers,
            use_res=self.use_res,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            attn_ffn_hidden_scale=attn_ffn_hidden_scale,
        )

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.to(device)


    # def update_edges(self):
    #     self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch


    # def get_edges(self):
    #     rows, cols = [], []
    #     for i in range(self.n_nodes):
    #         for j in range(self.n_nodes):
    #             if i != j:
    #                 rows.append(i)
    #                 cols.append(j)
    #     edges = [rows, cols]
    #     return edges


    # def get_edges_batch(self, batch_size):
    #     edges = self.get_edges()
    #     edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    #     edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    #     if batch_size > 1:
    #         rows, cols = [], []
    #         for i in range(batch_size):
    #             rows.append(edges[0] + self.n_nodes * i)
    #             cols.append(edges[1] + self.n_nodes * i)
    #         edges = [torch.cat(rows), torch.cat(cols)]
    #     edges[0] = edges[0].to(self.device)
    #     edges[1] = edges[1].to(self.device)
    #     return edges, edge_attr


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
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        obs = self.local_tool.local_info_process(obs, self.local_module)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, available_actions.shape[-1])
            available_actions = check(available_actions).to(**self.tpdv)

        # edges = self.local_tool.forward_edges

        x = self.hie_model(obs, self.n_nodes)

        actions, action_log_probs = self.act(
            x, available_actions, deterministic
        )
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
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        obs = self.local_tool.local_info_process(obs, self.local_module)

        # edges = self.local_tool.eval_edges

        x = self.hie_model(obs, self.n_nodes)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            x,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )
        return action_log_probs, dist_entropy, action_distribution
