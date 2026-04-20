import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *
from harl.models.base.model_util import *
from harl.models.base.hierarchy import HAMA

class HamaCriticNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(HamaCriticNet, self).__init__()
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        # args for env
        self.env_name = args["env_name"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.n_nodes = args["num_agents"]
        obs_dim = int(cent_obs_shape[0] / self.n_nodes)
        self.equ_nf = args.get("equ_nf", 4)
        self.inv_nf = int(cent_obs_shape[0] / self.n_nodes - self.equ_nf)

        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        # edges used for collect data
        self.collect_edges, _ = self.get_edges_batch(self.n_threads)
        # edges used for training
        self.train_edges, _ = self.get_edges_batch(self.mini_batch_size)

        # args for network
        self.hidden_sizes = args["hidden_sizes"]
        self.n_cluster = args["n_cluster"]
        self.n_heads = args["n_heads"]
        self.current_pooling_plan = None
        # self.interaction_layer = args["interaction_layer"]
        # self.pooling_layer = args["pooling_layer"]
        # self.decoder_layer = args["decoder_layer"]
        self.n_layers = args["n_layers"]
        self.flat = args["flat"]
        self.use_res = args["use_res"]
        self.lp_loss = 0

        self.use_history = args["use_history"]
        self.windows_size = args["windows_size"]

        # 局部信息处理
        self.local_info_input = args["local_info_input"]
        self.local_info_output = args["local_info_output"]
        self.hpn_hidden = args["hpn_hidden"]
        self.local_mode = args["local_mode"]
        self.use_hpn = args["use_hpn"]
        self.local_tool = local_tool(self, args=args, net_type="egnn", id="critic")
        self.local_tool.dim_info_init(obs_dim)
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

        self.hie_model = HAMA(
            in_node_nf=self.local_tool.inv_nf_new + self.local_tool.equ_nf,
            hidden_nf=self.hidden_sizes[-1],
            n_cluster=self.n_cluster,
            n_layer=self.n_layers,
            use_res=self.use_res,
            n_heads=self.n_heads
        )
        self.critic_fc1 = nn.Linear(self.hidden_sizes[-1], 1)
        self.to(device)

    def get_edges(self):
        rows, cols = [], []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        return edges

    def get_edges_batch(self, batch_size):
        edges = self.get_edges()
        edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
        edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
        if batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + self.n_nodes * i)
                cols.append(edges[1] + self.n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        edges[0] = edges[0].to(self.device)
        edges[1] = edges[1].to(self.device)
        return edges, edge_attr

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = self.local_tool.trans_info2local_critic(cent_obs)
        if cent_obs.shape[0] == self.mini_batch_size:
            edges = self.train_edges
            batches = self.mini_batch_size
        else:
            edges = self.collect_edges
            batches = self.n_threads
        # [n_batches, share_obs_shape] -> [n_batches, n_agents, obs_shape]
        cent_obs = np.reshape(cent_obs, (-1, self.n_nodes, self.local_tool.equ_nf + self.local_tool.inv_nf_old))
        # [n_batches, n_agents, obs_shape] -> [n_batches * n_agents, obs_shape]
        cent_obs = cent_obs.reshape(-1, self.local_tool.equ_nf + self.local_tool.inv_nf_old)
        # if self.args['obs_mode'] == 'local_obs_mode':
        #     cent_obs = cent_obs[:, :-self.n_nodes]
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        cent_obs = self.local_tool.local_info_process(cent_obs, self.local_module)

        x = self.hie_model(cent_obs, edges, self.n_nodes)
        value = self.critic_fc1(torch.tanh(x))
        value = torch.reshape(value, (batches, -1))

        values = value.mean(dim=1)
        values = torch.reshape(values, (-1, 1))
        return values, rnn_states
