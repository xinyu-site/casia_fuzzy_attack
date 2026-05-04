import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

# -------- Discrete-action (DQN-style) --------
class LocalQDiscreteNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions, hidden=hidden)
    def forward(self, obs):
        return self.mlp(obs)  # (B, A)

# -------- Continuous-action (DDPG-style) --------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = MLP(obs_dim, act_dim, hidden=hidden)
    def forward(self, obs):
        return torch.tanh(self.mlp(obs))

class LocalQContinuousNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = MLP(obs_dim + act_dim, 1, hidden=hidden)
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,)

# -------- Shared GMF Q --------
class GroupMFQNet(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = MLP(x_dim, 1, hidden=hidden)
    def forward(self, mu_x):
        return self.mlp(mu_x).squeeze(-1)


class GroupAttention(nn.Module):
    """Group attention to compute λ_k for μ(x)=Σ_k λ_k x_k (HMF paper Eq.(9))."""

    def __init__(self, x_dim: int, hidden: int = 64):
        super().__init__()
        self.key = nn.Linear(x_dim, hidden)
        self.query = nn.Linear(x_dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B,N,x_dim), mask: (B,N) {0,1}
        q = self.query(x)
        k = self.key(x)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        qg = (q * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        h = torch.tanh(k + qg)
        logits = self.score(h).squeeze(-1)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        lam = F.softmax(logits, dim=1) * mask
        lam = lam / lam.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return lam
