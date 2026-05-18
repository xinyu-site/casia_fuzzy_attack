"""EGHN V Critic."""
import torch
from harl.models.value_function_models.egnn_v2_critic_net import EgnnV2CriticNet
from harl.algorithms.critics.v_critic import VCritic


class Egnnv2VCriticEval(VCritic):
    """V Critic.
    Critic that learns a V-function.
    """
    def __init__(self, args, cent_obs_space, state_dict ,device=torch.device("cpu")):
        super(Egnnv2VCriticEval, self).__init__(args, cent_obs_space, device)
        self.critic = EgnnV2CriticNet(args, self.share_obs_space, self.device)
        self.critic.load_state_dict(state_dict)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
