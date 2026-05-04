from copy import deepcopy
import torch
from harl.algorithms.critics.continuous_q_critic import ContinuousQCritic
from harl.models.value_function_models.egnn_q_net import EgnnQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm

class EgnnQCritic(ContinuousQCritic):
    def __init__(self, args, share_obs_space, act_space, num_agents, state_type, device=torch.device("cpu")):
        super().__init__(args, share_obs_space, act_space, num_agents, state_type, device)
        self.critic = EgnnQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
        self.turn_off_grad()

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Train the critic.
        Args:
            share_obs: (np.ndarray) shape is (batch_size, dim)
            actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            reward: (np.ndarray) shape is (batch_size, 1)
            done: (np.ndarray) shape is (batch_size, 1)
            term: (np.ndarray) shape is (batch_size, 1)
            next_share_obs: (np.ndarray) shape is (batch_size, dim)
            next_actions: (np.ndarray) shape is (batch_size * n_agents, dim)
            gamma: (np.ndarray) shape is (batch_size, 1)
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        # actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_actions = check(next_actions).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_q_values = self.target_critic(next_share_obs, next_actions)

        train_info = {}
        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(self.critic(share_obs, actions), q_targets)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_max_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())
        train_info["critic_grad_norm"] += critic_grad_norm
        self.critic_optimizer.step()
        train_info["value_loss"] += critic_loss.item()
        return train_info