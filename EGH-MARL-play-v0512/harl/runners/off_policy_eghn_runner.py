"""Runner for off-policy MA algorithms"""
import copy
import torch
import numpy as np
import itertools
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
from harl.utils.models_tools import get_grad_norm


class OffPolicyEghnRunner(OffPolicyBaseRunner):
    """Runner for off-policy MA algorithms."""
    def __init__(self, args, algo_args, env_args):
        super(OffPolicyEghnRunner, self).__init__(args, algo_args, env_args)
        self.lp_coef = algo_args['algo']["lp_coef"]
        self.use_max_grad_norm = algo_args["algo"]["use_max_grad_norm"]
        self.max_grad_norm = algo_args["algo"]["max_grad_norm"]

    def train(self):
        """Train the model"""
        actor_train_info = []

        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()

        # size = sp_next_obs.shape[1]
        # temp = np.concatenate(sp_next_obs, axis=0)
        temp = np.concatenate(sp_next_obs.transpose(1, 0, 2), axis=0)
        next_actions = self.actor[0].get_target_actions(temp)
        # sp_actions = np.concatenate(sp_actions.transpose(1, 0, 2))
        # next_actions = [next_actions[i::self.num_agents] for i in range(self.num_agents)]

        critic_train_info = self.critic.train(
            sp_share_obs,
            np.concatenate(sp_actions.transpose(1, 0, 2)),
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
        )
        self.critic.turn_off_grad()
        if self.total_it % self.policy_freq == 0:
            # train actors
            # actions shape: (n_agents, batch_size, dim)
            train_info = {}
            train_info["policy_loss"] = 0
            train_info["actor_grad_norm"] = 0

            batch_size = sp_share_obs.shape[0]
            # mask_temp = []
            act_dim = self.action_spaces[0].shape[0]
            mask_temp = np.zeros((self.num_agents, act_dim), dtype=np.float32)
            # for i in range(self.num_agents):
                # mask_temp.append(np.zeros(act_dim, dtype=np.float32))
            mask = []
            for i in range(self.num_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                curr_mask_temp[i] = np.ones(act_dim, dtype=np.float32)
                # curr_mask_vec = np.concatenate(curr_mask_temp)
                curr_mask = np.tile(curr_mask_temp, (batch_size, 1))
                mask.append(curr_mask)
            mask = torch.tensor(np.concatenate(mask))

            self.actor[0].turn_on_grad()
            # train the agent
            pol_acts = self.actor[0].get_actions(np.concatenate(sp_obs.transpose(1, 0, 2)), False)
            actor_cent_acts = pol_acts.reshape(batch_size, self.num_agents, -1).reshape(batch_size, -1)
            # actor_cent_acts = pol_acts.split(split_size=batch_size, dim=0)
            # actor_cent_acts = [pol_acts[i::self.num_agents] for i in range(self.num_agents)]
            actor_cent_acts = pol_acts.repeat((self.num_agents, 1))
            buffer_cent_acts = torch.tensor(np.concatenate(sp_actions.transpose(1, 0, 2), axis=0)).repeat((self.num_agents, 1))
            update_cent_acts = mask * actor_cent_acts.cpu() + (1 - mask) * buffer_cent_acts
            stacked_cent_obs = np.tile(sp_share_obs, (self.num_agents, 1))

            value_pred = self.critic.get_values(stacked_cent_obs, update_cent_acts)
            actor_loss = -torch.mean(value_pred)
            self.critic.critic_optimizer.zero_grad()
            self.actor[0].actor_optimizer.zero_grad()
            actor_loss.backward()
            train_info["policy_loss"] += actor_loss.item()
            if self.use_max_grad_norm:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor[0].actor.parameters(), self.max_grad_norm
                )
            train_info["actor_grad_norm"] += actor_grad_norm
            self.actor[0].actor_optimizer.step()
            self.actor[0].turn_off_grad()
            self.actor[0].soft_update()
            self.critic.soft_update()

            # for _ in range(self.num_agents):
            actor_train_info.append(train_info)
        return actor_train_info, critic_train_info