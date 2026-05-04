"""HADDPG algorithm."""
from copy import deepcopy
import torch
from harl.models.policy_models.deterministic_egnn_policy import DeterministicEgnnPolicy
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.algorithms.actors.haddpg import HADDPG


class EGNNDDPG(HADDPG):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(EGNNDDPG, self).__init__(args, obs_space, act_space, device)
        self.actor = DeterministicEgnnPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()
