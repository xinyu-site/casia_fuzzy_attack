"""Runner for off-policy COMA algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
from harl.utils.models_tools import get_grad_norm
from harl.utils.trans_tools import _t2n


class OffPolicyCOMARunner(OffPolicyBaseRunner):
    """Runner for off-policy COMA algorithm.
    
    COMA (Counterfactual Multi-Agent Policy Gradients) is an off-policy algorithm
    that uses a centralized critic to estimate counterfactual advantages for
    decentralized actors.
    """

    def train(self):
        """Train the model for COMA algorithm.
        
        Returns:
            actor_train_infos: List of dictionaries containing actor training info for each agent
            critic_train_info: Dictionary containing critic training info
        """
        actor_train_infos = []
        critic_train_info = {}

        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
        ) = data

        # Prepare sample data for COMA
        # COMA expects specific format with obs, share_obs, actions, actions_onehot, etc.
        sample = {
            "obs": sp_obs,
            "share_obs": sp_share_obs,
            "actions": sp_actions,
            "actions_onehot": sp_actions,
            "rewards": sp_reward,
            "gammas": sp_gamma,
            "next_obs": sp_next_obs,
            "next_share_obs": sp_next_share_obs,
            "dones_env": sp_done,
            "masks": sp_valid_transition,
            "active_masks": sp_valid_transition,
            "filled": sp_valid_transition,
        }
        if sp_available_actions is not None:
            sample["available_actions"] = sp_available_actions
            sample["next_available_actions"] = sp_next_available_actions

        # Train critic
        # COMA critic computes Q-values for all actions
        self.critic.turn_on_grad()
        q_values = self.critic(sample)
        
        # Get target Q-values for TD learning
        with torch.no_grad():
            target_q_values = self.critic(sample)
        
        # Compute TD targets
        q_taken = torch.gather(q_values, dim=2, index=sp_actions.long()).squeeze(2)
        target_q_taken = torch.gather(target_q_values, dim=2, index=sp_actions.long()).squeeze(2)
        
        td_error = q_taken - target_q_taken.detach()
        critic_loss = (td_error ** 2).mean()
        
        # Optimize critic
        self.critic.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.algo_args["train"].get("use_max_grad_norm", False):
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.algo_args["train"].get("max_grad_norm", 0.5)
            ).item()
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())
        self.critic.critic_optimizer.step()
        self.critic.turn_off_grad()
        
        critic_train_info["critic_loss"] = critic_loss.item()
        critic_train_info["critic_grad_norm"] = critic_grad_norm
        critic_train_info["q_values"] = q_values.mean().item()
        
        # Train actors
        # COMA uses counterfactual baseline computed from the critic
        if self.total_it % self.policy_freq == 0:
            if self.share_param:
                # Shared parameter case: train single actor for all agents
                actor_loss, actor_grad_norm = self._train_shared_actor(sample, q_values)
                actor_train_infos.append({
                    "actor_loss": actor_loss.item(),
                    "actor_grad_norm": actor_grad_norm
                })
                # Duplicate for all agents
                for _ in range(self.num_agents - 1):
                    actor_train_infos.append(actor_train_infos[0])
            else:
                # Independent actors case
                for agent_id in range(self.num_agents):
                    actor_loss, actor_grad_norm = self._train_independent_actor(
                        agent_id, sample, q_values
                    )
                    actor_train_infos.append({
                        "actor_loss": actor_loss.item(),
                        "actor_grad_norm": actor_grad_norm
                    })
        else:
            # No actor update this iteration
            for agent_id in range(self.num_agents):
                actor_train_infos.append({
                    "actor_loss": 0.0,
                    "actor_grad_norm": 0.0
                })

        # Soft update target networks
        if self.share_param:
            self.actor[0].soft_update()
        else:
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
        self.critic.soft_update()

        return actor_train_infos, critic_train_info

    def _train_shared_actor(self, sample, q_values):
        """Train shared actor for all agents.
        
        Args:
            sample: Dictionary containing batch data
            q_values: Q-values from critic
            
        Returns:
            actor_loss: Scalar actor loss
            actor_grad_norm: Gradient norm of actor
        """
        self.actor[0].turn_on_grad()
        
        obs = sample["obs"]
        available_actions = sample.get("available_actions", None)
        actions = sample["actions"]
        active_masks = sample["active_masks"]
        
        # Get action distributions from actor
        action_dists = []
        for agent_id in range(self.num_agents):
            action_dist, _ = self.actor[0](
                obs[:, agent_id],
                None,  # rnn_states not used in off-policy
                active_masks[:, agent_id],
                available_actions[:, agent_id] if available_actions is not None else None
            )
            action_dists.append(action_dist)
        
        # Compute COMA advantage for each agent
        actor_loss = 0
        for agent_id in range(self.num_agents):
            # Get log probabilities
            log_probs = action_dists[agent_id].log_prob(actions[:, agent_id].squeeze(-1))
            
            # Compute counterfactual baseline
            # Baseline is the expected Q-value under current policy
            pi = action_dists[agent_id].probs
            q_vals_agent = q_values[:, agent_id, :]
            baseline = (pi * q_vals_agent).sum(dim=-1, keepdim=True).detach()
            
            # Compute advantage
            q_taken = torch.gather(q_vals_agent, dim=1, index=actions[:, agent_id].long())
            advantage = (q_taken - baseline).detach()
            
            # Compute policy gradient loss
            policy_loss = -(advantage * log_probs * active_masks[:, agent_id]).sum() / active_masks[:, agent_id].sum()
            actor_loss += policy_loss
        
        # Optimize actor
        self.actor[0].actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.algo_args["train"].get("use_max_grad_norm", False):
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor[0].parameters(),
                self.algo_args["train"].get("max_grad_norm", 0.5)
            ).item()
        else:
            actor_grad_norm = get_grad_norm(self.actor[0].parameters())
        self.actor[0].actor_optimizer.step()
        self.actor[0].turn_off_grad()
        
        return actor_loss / self.num_agents, actor_grad_norm

    def _train_independent_actor(self, agent_id, sample, q_values):
        """Train independent actor for a specific agent.
        
        Args:
            agent_id: ID of the agent to train
            sample: Dictionary containing batch data
            q_values: Q-values from critic
            
        Returns:
            actor_loss: Scalar actor loss
            actor_grad_norm: Gradient norm of actor
        """
        self.actor[agent_id].turn_on_grad()
        
        obs = sample["obs"]
        available_actions = sample.get("available_actions", None)
        actions = sample["actions"]
        active_masks = sample["active_masks"]
        
        # Get action distribution from actor
        action_dist, _ = self.actor[agent_id](
            obs[:, agent_id],
            None,  # rnn_states not used in off-policy
            active_masks[:, agent_id],
            available_actions[:, agent_id] if available_actions is not None else None
        )
        
        # Get log probabilities
        log_probs = action_dist.log_prob(actions[:, agent_id].squeeze(-1))
        
        # Compute counterfactual baseline
        pi = action_dist.probs
        q_vals_agent = q_values[:, agent_id, :]
        baseline = (pi * q_vals_agent).sum(dim=-1, keepdim=True).detach()
        
        # Compute advantage
        q_taken = torch.gather(q_vals_agent, dim=1, index=actions[:, agent_id].long())
        advantage = (q_taken - baseline).detach()
        
        # Compute policy gradient loss
        actor_loss = -(advantage * log_probs * active_masks[:, agent_id]).sum() / active_masks[:, agent_id].sum()
        
        # Optimize actor
        self.actor[agent_id].actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.algo_args["train"].get("use_max_grad_norm", False):
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor[agent_id].parameters(),
                self.algo_args["train"].get("max_grad_norm", 0.5)
            ).item()
        else:
            actor_grad_norm = get_grad_norm(self.actor[agent_id].parameters())
        self.actor[agent_id].actor_optimizer.step()
        self.actor[agent_id].turn_off_grad()
        
        return actor_loss, actor_grad_norm