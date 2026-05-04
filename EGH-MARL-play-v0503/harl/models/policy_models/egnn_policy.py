import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check
from torch.distributions.normal import Normal
from harl.models.base.egnn_clean import E_GCL
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space

class EgnnPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(EgnnPolicy, self).__init__()
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.hidden_nf = args["hidden_sizes"]
        self.n_nodes = args["num_agents"]
        self.equ_nf = args["equ_nf"]
        self.inv_nf = obs_space.shape[0] - self.equ_nf
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.n_layers = args["n_layers"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        self.forward_edges, _ = self.get_edges_batch(self.n_threads)
        self.eval_edges, _ = self.get_edges_batch(self.mini_batch_size)

        act_dim = get_shape_from_act_space(action_space)
        in_edge_nf = 0
        act_fn = nn.SiLU()
        residual = True
        attention = args["attention"]
        normalize = True
        tanh = True
    
        self.embedding_in = nn.Linear(self.inv_nf, self.hidden_nf[0])
        for i in range(0, self.n_layers):
            self.add_module("actor_gcl_%d" % i, E_GCL(self.hidden_nf[0], self.hidden_nf[0], self.hidden_nf[1], edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, initialization_method=self.initialization_method))
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.to(device)
        
    def update_edges(self):
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        self.forward_edges, _ = self.get_edges_batch(self.n_threads)
        self.eval_edges, _ = self.get_edges_batch(self.mini_batch_size)

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
        # import pdb;pdb.set_trace()
        obs = obs.reshape(-1, self.inv_nf + self.equ_nf)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        x = obs[:, :self.equ_nf]
        h = obs[:, self.equ_nf:]
        h = self.embedding_in(h)
        for j in range(0, self.n_layers):
            h, x, _ = self._modules["actor_gcl_%d" % j](h, self.forward_edges, x)
        mu = x

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
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        x = obs[:, :self.equ_nf]
        h = obs[:, self.equ_nf:]
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["actor_gcl_%d" % i](h, self.eval_edges, x)
        mu = x

        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        action_distribution = policy
        dist_entropy = action_distribution.entropy().sum(axis=-1)
        dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        action_log_probs = policy.log_prob(action)
        return action_log_probs, dist_entropy, action_distribution

