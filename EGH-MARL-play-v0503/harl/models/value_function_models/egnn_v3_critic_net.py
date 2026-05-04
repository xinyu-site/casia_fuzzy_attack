import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.base.eghn_v2 import EGNNv2
from harl.models.base.eghn_v2 import EGAN
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *
from harl.models.base.model_util import *

class EgnnV3CriticNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(EgnnV3CriticNet, self).__init__()
        self.env_name = args["env_name"]
        self.hidden_nf = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        
        # env args
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.n_nodes = args["num_agents"]
        self.int_points_num = args["int_points_num"]
        obs_dim = int(cent_obs_shape[0] / self.n_nodes)
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.critic_num_mini_batch
        self.dimension = args["dimension"]
        # edges used for collect data
        self.collect_edges, _ = self.get_edges_batch(self.n_threads)
        # edges used for training
        self.train_edges, _ = self.get_edges_batch(self.mini_batch_size)

        self.use_history = args["use_history"]
        self.windows_size = args["windows_size"]
        self.c = 1
        
        self.flat = args["flat"]
        self.in_edge_nf = args["in_edge_nf"]
        self.n_layers = args["n_layers"]
        
        # 局部信息处理
        self.local_info_input = args["local_info_input"]
        self.local_info_output = args["local_info_output"]
        self.hpn_hidden = args["hpn_hidden"]
        self.local_mode = args["local_mode"]
        self.use_hpn = args["use_hpn"]
        self.local_tool = local_tool(self, args=args, net_type="egnn", id="critic")
        self.local_tool.dim_info_init(obs_dim)
        self.n_vector_input = self.local_tool.equ_nf // self.dimension

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
            
        self.egnn_model = EGNNv2(
            n_layers=self.n_layers, 
            in_edge_nf=self.in_edge_nf,
            in_node_nf=self.local_tool.inv_nf_new, 
            hidden_nf=self.hidden_nf[0], 
            n_vector_input=self.n_vector_input,
            device=device,
            flat=self.flat, 
            activation=nn.SiLU(), 
            norm=True
        )

        # self.egnn_model = EGAN(
        #     n_layers=self.n_layers, 
        #     in_edge_nf=self.in_edge_nf,
        #     in_node_nf=self.local_tool.inv_nf_new, 
        #     hidden_nf=self.hidden_nf[0], 
        #     n_vector_input=self.n_vector_input,
        #     num_heads=2,
        #     embed_size=128,
        #     device=device,
        #     flat=self.flat, 
        #     activation=nn.SiLU(), 
        #     norm=True
        # )

        self.critic_fc1 = nn.Linear(self.n_vector_input * self.n_vector_input + self.hidden_nf[0], 1)
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
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
        
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        cent_obs = self.local_tool.local_info_process(cent_obs, self.local_module)
        
        equ_fea = cent_obs[:, :self.local_tool.equ_nf]
        h = cent_obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :self.dimension]
        # vel = equ_fea[:, 2:]
        equ_fea = equ_fea.reshape(equ_fea.shape[0], -1, self.dimension).transpose(1, 2)
        rows, cols = edges
        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        equ_out, h = self.egnn_model(equ_fea, h, edges, edge_fea=None)
        
        x = torch.matmul(equ_out.transpose(1, 2), equ_out)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, h), dim=1)
        x = torch.tanh(x)
        value = self.critic_fc1(x)
        value = torch.reshape(value, (batches, -1))
        values = value.mean(dim=1)
        values = torch.reshape(values, (-1, 1))
        return values, rnn_states
    
    
