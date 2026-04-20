"""Runner for off-policy MA algorithms"""
import copy
import torch
import numpy as np
import itertools
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
from harl.utils.models_tools import get_grad_norm


class OffPolicyMARunner(OffPolicyBaseRunner):
    """Runner for off-policy MA algorithms."""

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
        if self.flag:
            size = sp_next_obs.shape[1]
            temp = np.concatenate(sp_next_obs, axis=0)
            next_actions = self.actor[0].get_target_actions(temp)
            next_actions = next_actions.cpu().split(split_size=size, dim=0)
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
        critic_train_info = self.critic.train(
            sp_share_obs,
            sp_actions,
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
            if self.flag:
                batch_size = sp_share_obs.shape[0]
                mask_temp = []
                act_dim = self.action_spaces[0].shape[0]
                for i in range(self.num_agents):
                    mask_temp.append(np.zeros(act_dim, dtype=np.float32))
                mask = []
                for i in range(self.num_agents):
                    curr_mask_temp = copy.deepcopy(mask_temp)
                    curr_mask_temp[i] = np.ones(act_dim, dtype=np.float32)
                    curr_mask_vec = np.concatenate(curr_mask_temp)
                    curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                    mask.append(curr_mask)
                mask = torch.tensor(np.concatenate(mask))

                self.actor[0].turn_on_grad()
                # train the agent
                pol_acts = self.actor[0].get_actions(np.concatenate(sp_obs), False)
                actor_cent_acts = pol_acts.split(split_size=batch_size, dim=0)
                actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((self.num_agents, 1))
                buffer_cent_acts = torch.tensor(np.concatenate(sp_actions, axis=-1)).repeat((self.num_agents, 1))
                update_cent_acts = mask * actor_cent_acts.cpu() + (1 - mask) * buffer_cent_acts
                stacked_cent_obs = np.tile(sp_share_obs, (self.num_agents, 1))

                value_pred = self.critic.get_values(stacked_cent_obs, update_cent_acts)
                actor_loss = -torch.mean(value_pred)
                self.critic.critic_optimizer.zero_grad()
                self.actor[0].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[0].actor_optimizer.step()
                self.actor[0].turn_off_grad()
                self.actor[0].soft_update()
                train_info["policy_loss"] += actor_loss.item()
                train_info["actor_grad_norm"] += get_grad_norm(self.actor[0].actor.parameters())
                actor_train_info.append(train_info)
            else:
                for agent_id in range(self.num_agents):
                    actions = copy.deepcopy(torch.tensor(sp_actions)).to(self.device)
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id] = self.actor[agent_id].get_actions(
                        sp_obs[agent_id], False
                    )
                    actions_list = [a for a in actions]
                    actions_t = torch.cat(actions_list, dim=-1)
                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    actor_loss = -torch.mean(value_pred)
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                # soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
        return actor_train_info, critic_train_info
