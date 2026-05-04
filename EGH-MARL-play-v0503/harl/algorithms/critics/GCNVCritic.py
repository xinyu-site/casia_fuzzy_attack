"""EGHN V Critic."""
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.gcn_critic_net import GCNCriticNet
from harl.algorithms.critics.v_critic import VCritic


class GCNVCritic(VCritic):
    """V Critic.
    Critic that learns a V-function.
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(GCNVCritic, self).__init__(args, cent_obs_space, device)
        self.critic = GCNCriticNet(args, self.share_obs_space, self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
