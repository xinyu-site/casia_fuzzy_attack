import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.model_util import *

# # 控制RNN使用EQC时对于rnn_states是采用mean还是直接采用恒等变换的states
# rnn_mean_flag = False

class StochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.equ_nf = args["equ_nf"]
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_eqc_flag = args["use_eqc_flag"]
        self.subgroup_num = int(args["subgroup_num"])
        self.rnn_mean_flag = args["rnn_mean_flag"]
        # 时序信息处理
        self.use_history = args["use_history"]
        self.windows_size = args["windows_size"]
        self.c = 1

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

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
        if self.use_eqc_flag:
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_obs_list = []
            for i in range(self.subgroup_num):
                angle = i * angle_increment
                if i == 0:
                    rotated_obs = obs.copy()
                else:
                    rotated_obs = rotation_obs3d(obs.copy(), angle, self.equ_nf)
                rotated_obs_list.append(rotated_obs)
            obs = np.concatenate(rotated_obs_list, axis=0)
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            if self.use_eqc_flag:
                rnn_states = rnn_states.repeat(1, self.subgroup_num, 1, 1)
                masks = masks.repeat(1, self.subgroup_num, 1)
            if len(rnn_states.shape) != 3:
                actor_features_list = []
                rnn_states_list = []
                for i in range(rnn_states.shape[0]):
                    actor_feature, rnn_state = self.rnn(actor_features[:, i, :], rnn_states[i], masks[i])
                    actor_features_list.append(actor_feature)
                    rnn_states_list.append(rnn_state)
                actor_features = torch.stack(actor_features_list, dim=1)
                rnn_states = torch.stack(rnn_states_list)
            else:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            if self.use_eqc_flag:
                parael_num = rnn_states.shape[1]//self.subgroup_num
                rnn_states_list = []
                for i in range(self.subgroup_num):
                    start_idx = parael_num * i
                    end_idx = parael_num * (i + 1)
                    rnn_state = rnn_states[:, start_idx:end_idx, :, :]
                    rnn_states_list.append(rnn_state)
                if self.rnn_mean_flag:
                    stacked_tensor = torch.stack(rnn_states_list, dim=0)
                    rnn_states = stacked_tensor.mean(dim=0)
                else:
                    rnn_states = rnn_states_list[0]

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        
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
        if self.use_eqc_flag:
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_obs_list = []
            for i in range(self.subgroup_num):
                angle = i * angle_increment
                if i == 0:
                    rotated_obs = obs.copy()
                else:
                    rotated_obs = rotation_obs3d(obs.copy(), angle, self.equ_nf)
                rotated_obs_list.append(rotated_obs)
            obs = np.concatenate(rotated_obs_list, axis=0)
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            if self.use_eqc_flag:
                rnn_states = rnn_states.repeat(1, self.subgroup_num, 1, 1)
                masks = masks.repeat(1, self.subgroup_num, 1)
            if len(rnn_states.shape) != 3:
                actor_features_list = []
                rnn_states_list = []
                for i in range(rnn_states.shape[0]):
                    actor_feature, rnn_state = self.rnn(actor_features[:, i, :], rnn_states[i], masks[i])
                    actor_features_list.append(actor_feature)
                    rnn_states_list.append(rnn_state)
                actor_features = torch.stack(actor_features_list, dim=1)
                rnn_states = torch.stack(rnn_states_list)
            else:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            if self.use_eqc_flag:
                parael_num = rnn_states.shape[1]//self.subgroup_num
                rnn_states_list = []
                for i in range(self.subgroup_num):
                    start_idx = parael_num * i
                    end_idx = parael_num * (i + 1)
                    rnn_state = rnn_states[:, start_idx:end_idx, :, :]
                    rnn_states_list.append(rnn_state)
                if self.rnn_mean_flag:
                    stacked_tensor = torch.stack(rnn_states_list, dim=0)
                    rnn_states = stacked_tensor.mean(dim=0)
                else:
                    rnn_states = rnn_states_list[0]

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        
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
        if self.use_eqc_flag:
            angle_increment = 2 * np.pi / self.subgroup_num
            rotated_obs_list = []
            for i in range(self.subgroup_num):
                angle = i * angle_increment
                if i == 0:
                    rotated_obs = obs.copy()
                else:
                    rotated_obs = rotation_obs2d(obs.copy(), angle, self.equ_nf)
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

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            if self.use_eqc_flag:
                rnn_states = rnn_states.repeat(self.subgroup_num, 1, 1)
                masks = masks.repeat(self.subgroup_num, 1)
            if len(rnn_states.shape) != 3:
                actor_features_list = []
                rnn_states_list = []
                for i in range(rnn_states.shape[0]):
                    actor_feature, rnn_state = self.rnn(actor_features[i], rnn_states[i], masks[i])
                    actor_features_list.append(actor_feature)
                    rnn_states_list.append(rnn_state)
                actor_features = torch.stack(actor_features_list)
                rnn_states = torch.stack(rnn_states_list)
            else:            
                actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            if self.use_eqc_flag:
                parael_num = rnn_states.shape[0]//self.subgroup_num
                rnn_states_list = []
                for i in range(self.subgroup_num):
                    start_idx = parael_num * i
                    end_idx = parael_num * (i + 1)
                    rnn_state = rnn_states[start_idx:end_idx, :, :]
                    rnn_states_list.append(rnn_state)
                if self.rnn_mean_flag:
                    stacked_tensor = torch.stack(rnn_states_list, dim=0)
                    rnn_states = stacked_tensor.mean(dim=0)
                else:
                    rnn_states = rnn_states_list[0]

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )
            
        return action_log_probs, dist_entropy, action_distribution
