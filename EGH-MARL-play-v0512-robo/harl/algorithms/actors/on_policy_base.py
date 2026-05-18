"""Base class for on-policy algorithms."""

import torch
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.models_tools import (
    update_linear_schedule, 
    update_exponential_decay, 
    update_polynomial_decay, 
    update_customized_decay,
    update_cosine_decay,
    update_step_decay
)

class OnPolicyBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """初始化基于策略的算法基类。
        Args:
            args: (dict) 算法参数字典，包含所有超参数配置
            obs_space: (gym.spaces or list) 观测空间，定义智能体的输入维度
            act_space: (gym.spaces) 动作空间，定义智能体的输出动作范围
            device: (torch.device) 计算设备，用于张量运算（CPU或GPU）
        """
        # 保存基本参数
        self.args = args
        self.device = device
        # 创建张量设备配置字典，用于后续数据类型和设备转换
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 数据分块长度，用于RNN策略的数据分块处理
        self.data_chunk_length = args["data_chunk_length"]
        # 是否使用循环神经网络策略
        self.use_recurrent_policy = args["use_recurrent_policy"]
        # 是否使用朴素循环神经网络策略
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        # 是否使用策略激活掩码，用于处理死亡智能体
        self.use_policy_active_masks = args["use_policy_active_masks"]
        # 动作聚合方式，用于多智能体协作
        self.action_aggregation = args["action_aggregation"]

        # 学习率，控制优化器步长
        self.lr = args["lr"]
        # 优化器数值稳定性参数，防止除零错误
        self.opti_eps = args["opti_eps"]
        # 权重衰减系数，用于L2正则化，防止过拟合
        self.weight_decay = args["weight_decay"]
        # 保存观测空间和动作空间，用于后续网络构建
        self.obs_space = obs_space
        self.act_space = act_space
        # 创建随机策略网络，作为智能体的决策模型
        self.actor = StochasticPolicy(args, self.obs_space, self.act_space, self.device)
        # 创建Adam优化器，用于策略网络的参数更新
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes, decay_mode="linear"):
        """Decay the learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        if decay_mode == "linear":
            update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        elif decay_mode == "exponential":
            update_exponential_decay(self.actor_optimizer, episode,episodes, self.lr)
        elif decay_mode == "polynomial":
            update_polynomial_decay(self.actor_optimizer, episode,episodes, self.lr)
        elif decay_mode == "customized":
            update_customized_decay(self.actor_optimizer, episode,episodes, self.lr)
        elif decay_mode == "cosine":
            update_cosine_decay(self.actor_optimizer, episode,episodes, self.lr)
        elif decay_mode == "step":
            update_step_decay(self.actor_optimizer, episode,episodes, self.lr)
        else:
            raise Exception("no such decay mode!")

    # 获取动作
    def get_actions(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states_actor

    # 评估动作
    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray / torch.Tensor) if actor has RNN layer, RNN states for actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            masks: (np.ndarray / torch.Tensor) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        """

        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        if deterministic==False:
            actions, _, rnn_states_actor = self.actor(
                obs, rnn_states_actor, masks, available_actions, deterministic
            )
        else:
            actions, rnn_states_actor = self.actor.forward_decided(
                obs, rnn_states_actor, masks, available_actions, deterministic
            )
        return actions, rnn_states_actor

    def act_grd(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, rnn_states_actor = self.actor.forward_grd(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor

    # 更新网络
    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        pass

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        """
        pass

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.actor.eval()