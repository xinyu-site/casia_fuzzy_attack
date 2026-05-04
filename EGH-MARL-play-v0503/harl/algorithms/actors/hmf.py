"""HMF actor wrapper for HARL.

This actor is used only for action selection during rollout.
The actual optimization is handled by HMFAgentSystem in OffPolicyHMFRunner.
"""
from copy import deepcopy
import numpy as np
import torch
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.models.value_function_models.hmf_q_net import LocalQDiscreteNet, PolicyNet


class HMF(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args.get("polyak", 0.01)
        self.lr = args.get("lr", 3e-4)
        self.epsilon = args.get("epsilon", 0.1)
        self.is_discrete = act_space.__class__.__name__ == "Discrete"
        obs_dim = obs_space.shape[0]

        if self.is_discrete:
            self.action_dim = act_space.n
            self.actor = LocalQDiscreteNet(obs_dim, self.action_dim).to(device)
            self.target_actor = deepcopy(self.actor)
            for p in self.target_actor.parameters():
                p.requires_grad = False
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        else:
            self.action_dim = act_space.shape[0]
            self.act_low = torch.as_tensor(act_space.low, device=device, dtype=torch.float32)
            self.act_high = torch.as_tensor(act_space.high, device=device, dtype=torch.float32)
            self.actor = PolicyNet(obs_dim, self.action_dim).to(device)
            self.target_actor = deepcopy(self.actor)
            for p in self.target_actor.parameters():
                p.requires_grad = False
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.expl_noise = args.get("expl_noise", 0.1)

        self.turn_off_grad()

    def _scale_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return self.act_low + (a_tanh + 1.0) * 0.5 * (self.act_high - self.act_low)

    @staticmethod
    def _mask_q(q: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        """Mask q-values where avail==0 by setting to very negative."""
        return q.masked_fill(avail <= 0.0, -1e9)

    def get_actions(self, obs, available_actions=None, epsilon_greedy=True):
        obs = check(obs).to(**self.tpdv)

        if self.is_discrete:
            if available_actions is not None:
                avail = check(available_actions).to(**self.tpdv)  # (B,A)
            else:
                avail = None

            if np.random.random() < self.epsilon and epsilon_greedy:
                if avail is None:
                    actions = torch.randint(low=0, high=self.action_dim, size=(*obs.shape[:-1], 1))
                else:
                    # sample uniformly among available actions
                    probs = avail / avail.sum(dim=-1, keepdim=True).clamp(min=1.0)
                    actions = torch.multinomial(probs, num_samples=1)
                return actions

            q = self.actor(obs)  # (B,A)
            if avail is not None:
                q = self._mask_q(q, avail)
            return q.argmax(dim=-1, keepdim=True)

        # continuous
        a = self._scale_action(self.actor(obs))
        if epsilon_greedy:
            a = a + torch.randn_like(a) * float(self.expl_noise)
            a = torch.max(torch.min(a, self.act_high), self.act_low)
        return a

    def get_target_actions(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        if self.is_discrete:
            q = self.target_actor(obs)
            if available_actions is not None:
                avail = check(available_actions).to(**self.tpdv)
                q = self._mask_q(q, avail)
            return q.argmax(dim=-1, keepdim=True)

        a = self._scale_action(self.target_actor(obs))
        a = torch.max(torch.min(a, self.act_high), self.act_low)
        return a
