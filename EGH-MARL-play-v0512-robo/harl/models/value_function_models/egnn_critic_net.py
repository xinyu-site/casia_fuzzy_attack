import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import get_init_method
from harl.models.base.egnn_clean import E_GCL

class EgnnCriticNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(EgnnCriticNet, self).__init__()
        self.initialization_method = args["initialization_method"]
        # self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        # self.use_recurrent_policy = args["use_recurrent_policy"]
        # self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.device = device
        
        self.hidden_nf = args["hidden_sizes"]
        self.n_nodes = args["num_agents"]
        self.equ_nf = args["equ_nf"]
        self.inv_nf = int(cent_obs_shape[0] / self.n_nodes - self.equ_nf)
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.n_layers = args["n_layers"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.critic_num_mini_batch
        # edges used for collect data
        self.collect_edges, _ = self.get_edges_batch(self.n_threads)
        # edges used for training
        self.train_edges, _ = self.get_edges_batch(self.mini_batch_size)

        act_dim = 2
        in_edge_nf = 0
        act_fn = nn.SiLU()
        residual = True
        attention = args["attention"]
        normalize = True
        tanh = True

        self.embedding_in = nn.Linear(self.inv_nf, self.hidden_nf[0])
        for i in range(0, self.n_layers):
            self.add_module("critic_gcl_%d" % i, E_GCL(self.hidden_nf[0], self.hidden_nf[0], self.hidden_nf[1], edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, initialization_method=self.initialization_method))
        self.critic_fc1 = nn.Linear(1 + self.hidden_nf[0], self.hidden_nf[0])
        self.critic_fc2 = nn.Linear(self.hidden_nf[0], 1)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
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
        if cent_obs.shape[0] == self.mini_batch_size:
            edges = self.train_edges
            batches = self.mini_batch_size
        else:
            edges = self.collect_edges
            batches = self.n_threads
        # [n_batches, share_obs_shape] -> [n_batches, n_agents, obs_shape]
        cent_obs = np.reshape(cent_obs, (-1, self.n_nodes, self.equ_nf + self.inv_nf))
        # [n_batches, n_agents, obs_shape] -> [n_batches * n_agents, obs_shape]
        cent_obs = cent_obs.reshape(-1, self.equ_nf + self.inv_nf)
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        x = cent_obs[:, :self.equ_nf]
        h = cent_obs[:, self.equ_nf:]
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["critic_gcl_%d" % i](h, edges, x)
        x = torch.sum(x**2, dim=1, keepdim=True)
        x = torch.cat((x, h), dim=1)
        x = torch.tanh(self.critic_fc1(x))
        value = self.critic_fc2(x)

        value = torch.reshape(value, (batches, -1))
        # value = torch.mean(value, dim=0)
        # value_list.append(value)

        values = value.mean(dim=1)
        values = torch.reshape(values, (-1, 1))
        return values, rnn_states
