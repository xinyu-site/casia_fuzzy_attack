"""MAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.hepn_policy import HepnPolicy


class HepnMAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HepnMAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.lp_coef = args["actor_lp_coef"]
        self.actor = HepnPolicy(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_actions(
        self, obs, rnn_states_actor, masks, layer_data, available_actions=None, deterministic=False
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
            obs, rnn_states_actor, masks, layer_data, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states_actor

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        layer_data,
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
            obs, rnn_states_actor, action, masks, layer_data, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
        self, obs, rnn_states_actor, masks, layer_data, available_actions=None, deterministic=False
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
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, layer_data, available_actions, deterministic
        )
        return actions, rnn_states_actor

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            layer_data,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            layer_data,
            available_actions_batch,
            active_masks_batch,
        )
        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        # lp_loss = self.actor.lp_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def share_param_train(self, actor_buffer, graph_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        # train_info["lp_loss"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)
            graph_generator = graph_buffer.feed_forward_generator_actor(self.actor_num_mini_batch)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                layer_data = next(graph_generator)
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                adjust_batches = [[] for _ in range(8)]
                num = int(batches[0].shape[0]/10)
                for j in range(8):
                    if batches[j] is None:
                        adjust_batches[j] = None
                        continue
                    adjust_batches[j] = np.zeros_like(batches[j])
                    temp_obs = batches[j].copy()
                    for i in range(num):
                        adjust_batches[j][i*10:i*10+10] = temp_obs[i:i+num*10:num]
                    del temp_obs
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    tuple(adjust_batches) + (layer_data,)
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                # train_info["lp_loss"] += lp_loss.item()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
