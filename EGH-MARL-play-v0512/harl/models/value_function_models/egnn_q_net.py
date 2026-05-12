import torch
import numpy as np
import torch.nn as nn
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.egnn import EGNN


def get_combined_dim(act_spaces):
    """Get the combined dimension of individual actions."""
    combined_dim = 0
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


class EgnnQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """

    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(EgnnQNet, self).__init__()
        # activation_func = args["activation_func"]
        self.hidden_nf = args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        act_dim = act_spaces[0].shape[0]
        self.device = device
        
        self.n_nodes = args["num_agents"]
        self.n_threads = args["n_rollout_threads"]
        self.equ_nf = args["equ_nf"]
        self.inv_nf = int(cent_obs_shape[0] / self.n_nodes - self.equ_nf)
        self.batch_size = args["batch_size"]
        # edges used for collect data
        self.edges, _ = self.get_edges_batch(self.batch_size)
        # edges used for updating actor
        self.actor_edges, _ = self.get_edges_batch(self.batch_size * self.n_nodes)

        self.n_layers = args['n_layers']
        self.flat = args["flat"]
        self.in_edge_nf = args["in_edge_nf"]

        self.egnn_model = EGNN(n_layers=self.n_layers, in_edge_nf=self.in_edge_nf,in_node_nf=self.inv_nf, hidden_nf=self.hidden_nf[0], device=device,
                                flat=self.flat,  with_v=True, activation=nn.SiLU(), norm=True)
        self.critic_fc1 = nn.Linear(self.hidden_nf[-1], 1)
        # self.critic_fc2 = nn.Linear(hidden_sizes[-1], 1)
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

    def forward(self, cent_obs, actions):
        if cent_obs.shape[0] == self.batch_size:
            edges = self.edges
            batches = self.batch_size
        else:
            edges = self.actor_edges
            batches = self.batch_size * self.n_nodes
        # [n_batches, share_obs_shape] -> [n_batches, n_agents, obs_shape]
        cent_obs = cent_obs.reshape(batches, self.n_nodes, -1)
        # actions = actions.reshape(batches, self.n_nodes, -1)
        # [n_batches, n_agents, obs_shape] -> [n_batches * n_agents, obs_shape]
        cent_obs = cent_obs.reshape(batches * self.n_nodes, -1)
        # actions = actions.reshape(batches * self.n_nodes, -1)

        equ_fea = cent_obs[:, self.inv_nf:]
        inv_fea = cent_obs[:, :self.inv_nf]
        loc = equ_fea[:, :2]
        vel = equ_fea[:, 2:]
        # vel = equ_fea
        local_edge_index = edges
        # inv_fea = torch.cat((inv_fea, torch.sum(equ_fea**2, dim=1, keepdim=True)), dim=1)
        
        rows, cols = edges
        edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        local_edge_fea = edge_attr
        loc_pred, vel_pred, h = self.egnn_model(loc, inv_fea, edges, edge_attr, v=actions)
        # x = torch.sum(vel_pred**2, dim=1, keepdim=True)
        # a = torch.sum(actions**2, dim=1, keepdim=True)
        # x = torch.norm(vel_pred, p=2, dim=1, keepdim=True)
        # a = torch.norm(actions, p=2, dim=1, keepdim=True)
        # x = torch.cat((h, x, a), dim=1)
        # q_values = torch.tanh(self.critic_fc1(x))
        # q_values = self.critic_fc2(x)
        q_values = self.critic_fc1(torch.tanh(h))

        q_values = torch.reshape(q_values, (batches, -1))
        q_values = q_values.mean(dim=1, keepdim=True)

        return q_values
