import torch
import torch.nn as nn
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.models.base.egnn import EGNN


class DeterministicEgnnPolicy(nn.Module):
    """Deterministic policy network for continuous action space."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.hidden_nf = args["hidden_sizes"]
        # activation_func = args["activation_func"]
        # final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)

        self.device = device
        self.n_nodes = args["nr_agents"]
        self.n_threads = args["n_rollout_threads"]
        self.batch_size = args["batch_size"]
        self.collect_edges, _ = self.get_edges_batch(self.n_threads)
        self.train_edges, _ = self.get_edges_batch(self.batch_size)

        self.equ_nf = args["equ_nf"]
        self.inv_nf = obs_shape[0] - self.equ_nf
        self.n_layers = args["n_layers"]
        self.in_edge_nf = args["in_edge_nf"]

        self.flat = args["flat"]
        self.lp_loss = 0

        self.pi = EGNN(n_layers=self.n_layers, in_edge_nf=self.in_edge_nf,in_node_nf=self.inv_nf, hidden_nf=self.hidden_nf[0], device=device,
                        flat=self.flat,  with_v=True, activation=nn.SiLU(), norm=True)
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(**self.tpdv)
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
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

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        if obs.shape[0] == self.n_nodes * self.batch_size:
            # obs = obs.reshape(self.n_nodes, self.batch_size, -1)
            # obs = obs.transpose(0, 1)
            # obs = obs.reshape(self.batch_size * self.n_nodes, -1)
            edges = self.train_edges
        else:
            edges = self.collect_edges
        equ_fea = obs[:, self.inv_nf:]
        inv_fea = obs[:, :self.inv_nf]
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        local_edge_index = edges
        rows, cols = edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        local_edge_fea = edge_attr
        loc_pred, vel_pred, _ = self.pi(loc, inv_fea, edges, edge_attr, v=vel)
        x = vel_pred
        x = self.scale * x + self.mean
        return x
