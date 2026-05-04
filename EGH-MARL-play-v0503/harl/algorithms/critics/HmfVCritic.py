"""EGHN V Critic."""
import torch
from harl.models.value_function_models.hmf_critic import HmfCriticNet
from harl.algorithms.critics.v_critic import VCritic


class HmfVCritic(VCritic):
    """V Critic.
    Critic that learns a V-function.
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(HmfVCritic, self).__init__(args, cent_obs_space, device)
        self.critic = HmfCriticNet(args, self.share_obs_space, self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
