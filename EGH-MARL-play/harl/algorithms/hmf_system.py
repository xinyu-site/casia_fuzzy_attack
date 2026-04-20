import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.models.value_function_models.hmf_q_net import (
    LocalQDiscreteNet, 
    LocalQContinuousNet, 
    PolicyNet,
    GroupMFQNet, 
    GroupAttention
)

class HMFAgentSystem:
    """Hierarchical Mean-Field (HMF) learner supporting discrete & continuous actions.

    SMAC/SMACv2 notes (discrete):
      1) Must respect available_actions / next_available_actions masks when selecting argmax actions
         to avoid illegal actions and incorrect TD targets.
      2) Reward is often a *shared team reward* replicated per-agent (rew shape (B,N) but identical).
         In that case, summing over agents/groups scales rewards by N/group_size and destabilizes learning.
         This implementation treats reward as shared by default:
            r_global = rew.mean(dim=1)  (identical -> original scalar)
            use r_global for both group TD target and top TD target.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_space,
        n_groups: int,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.01,
        reg_alpha: float = 0.0,
        ipc_beta: float = 0.0,
        seed: int = 0,
        hidden: int = 128,
        attn_hidden: int = 64,
        actor_lr=None,
    ):
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.n_groups = int(n_groups)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.reg_alpha = float(reg_alpha)
        self.ipc_beta = float(ipc_beta)
        self.rng = np.random.default_rng(seed)

        self.is_discrete = act_space.__class__.__name__ == "Discrete"
        if self.is_discrete:
            self.n_actions = int(act_space.n)
            self.act_dim = None
            x_dim = self.obs_dim + self.n_actions
        else:
            self.act_dim = int(act_space.shape[0])
            self.n_actions = None
            x_dim = self.obs_dim + self.act_dim
            self.act_low = torch.as_tensor(act_space.low, device=self.device, dtype=torch.float32)
            self.act_high = torch.as_tensor(act_space.high, device=self.device, dtype=torch.float32)

        self.group_ids = np.arange(self.n_agents) % self.n_groups

        if self.is_discrete:
            self.local_q = LocalQDiscreteNet(self.obs_dim, self.n_actions, hidden=hidden).to(self.device)
            self.local_q_tgt = LocalQDiscreteNet(self.obs_dim, self.n_actions, hidden=hidden).to(self.device)
            self.local_q_tgt.load_state_dict(self.local_q.state_dict())
            self.actor = self.actor_tgt = None
            self.local_critic = self.local_critic_tgt = None
        else:
            self.actor = PolicyNet(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
            self.actor_tgt = PolicyNet(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
            self.actor_tgt.load_state_dict(self.actor.state_dict())

            self.local_critic = LocalQContinuousNet(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
            self.local_critic_tgt = LocalQContinuousNet(self.obs_dim, self.act_dim, hidden=hidden).to(self.device)
            self.local_critic_tgt.load_state_dict(self.local_critic.state_dict())

            self.local_q = self.local_q_tgt = None

        self.gmf_q = nn.ModuleList([GroupMFQNet(x_dim, hidden=hidden).to(self.device) for _ in range(self.n_groups)])
        self.gmf_q_tgt = nn.ModuleList([GroupMFQNet(x_dim, hidden=hidden).to(self.device) for _ in range(self.n_groups)])
        for i in range(self.n_groups):
            self.gmf_q_tgt[i].load_state_dict(self.gmf_q[i].state_dict())
        self.attn = nn.ModuleList([GroupAttention(x_dim, hidden=attn_hidden).to(self.device) for _ in range(self.n_groups)])

        self.mean_policy = None
        if self.is_discrete:
            self.mean_policy = torch.ones(self.n_groups, self.n_actions, device=self.device) / self.n_actions

        if self.is_discrete:
            params = list(self.local_q.parameters())
            for i in range(self.n_groups):
                params += list(self.gmf_q[i].parameters()) + list(self.attn[i].parameters())
            self.opt = torch.optim.Adam(params, lr=lr)
            self.actor_opt = None
        else:
            critic_params = list(self.local_critic.parameters())
            for i in range(self.n_groups):
                critic_params += list(self.gmf_q[i].parameters()) + list(self.attn[i].parameters())
            self.opt = torch.optim.Adam(critic_params, lr=lr)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr or lr))

    def _onehot(self, act: torch.Tensor) -> torch.Tensor:
        return F.one_hot(act, num_classes=self.n_actions).float()

    def _scale_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return self.act_low + (a_tanh + 1.0) * 0.5 * (self.act_high - self.act_low)

    @staticmethod
    def _mask_q(q: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        return q.masked_fill(avail <= 0.0, -1e9)

    def _build_x(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if self.is_discrete:
            return torch.cat([obs, self._onehot(act)], dim=-1)
        return torch.cat([obs, act], dim=-1)

    def _group_mask(self, group_idx: int, B: int) -> torch.Tensor:
        gids = torch.as_tensor(self.group_ids, device=self.device)
        mask1 = (gids == group_idx).float()
        return mask1.unsqueeze(0).repeat(B, 1)

    @staticmethod
    def _soft_update(tgt: nn.Module, src: nn.Module, tau: float):
        with torch.no_grad():
            for p_t, p in zip(tgt.parameters(), src.parameters()):
                p_t.data.mul_(1 - tau).add_(tau * p.data)

    def state_dict(self):
        sd = {
            "gmf_q": [m.state_dict() for m in self.gmf_q],
            "gmf_q_tgt": [m.state_dict() for m in self.gmf_q_tgt],
            "attn": [m.state_dict() for m in self.attn],
            "group_ids": self.group_ids,
            "is_discrete": self.is_discrete,
        }
        if self.is_discrete:
            sd.update({
                "local_q": self.local_q.state_dict(),
                "local_q_tgt": self.local_q_tgt.state_dict(),
                "mean_policy": self.mean_policy.detach().cpu(),
            })
        else:
            sd.update({
                "actor": self.actor.state_dict(),
                "actor_tgt": self.actor_tgt.state_dict(),
                "local_critic": self.local_critic.state_dict(),
                "local_critic_tgt": self.local_critic_tgt.state_dict(),
            })
        return sd

    def load_state_dict(self, sd):
        for i in range(self.n_groups):
            self.gmf_q[i].load_state_dict(sd["gmf_q"][i])
            self.gmf_q_tgt[i].load_state_dict(sd["gmf_q_tgt"][i])
            self.attn[i].load_state_dict(sd["attn"][i])
        self.group_ids = np.asarray(sd.get("group_ids", self.group_ids), dtype=np.int64)

        if sd.get("is_discrete", self.is_discrete):
            if self.local_q is not None:
                self.local_q.load_state_dict(sd["local_q"])
                self.local_q_tgt.load_state_dict(sd["local_q_tgt"])
                self.mean_policy = torch.as_tensor(sd["mean_policy"], device=self.device, dtype=torch.float32)
        else:
            if self.actor is not None:
                self.actor.load_state_dict(sd["actor"])
                self.actor_tgt.load_state_dict(sd["actor_tgt"])
                self.local_critic.load_state_dict(sd["local_critic"])
                self.local_critic_tgt.load_state_dict(sd["local_critic_tgt"])

    def update(
        self,
        obs,
        act,
        rew,
        next_obs,
        done,
        update_mean_policy: bool = True,
        available_actions=None,
        next_available_actions=None,
    ):
        """One gradient update.

        Shapes:
          obs:  (B,N,obs_dim)
          act:  Discrete (B,N) / Continuous (B,N,act_dim)
          rew:  (B,N) (often identical across N in SMACv2)
          next_obs: (B,N,obs_dim)
          done: (B,1) or (B,)
          available_actions / next_available_actions (discrete only): (B,N,A)
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1)
        B = obs.shape[0]

        # Shared/team reward (robust to per-agent replication)
        r_global = rew.mean(dim=1)  # (B,)

        if self.is_discrete:
            act = torch.as_tensor(act, dtype=torch.long, device=self.device)
            avail = None if available_actions is None else torch.as_tensor(available_actions, dtype=torch.float32, device=self.device)
            navail = None if next_available_actions is None else torch.as_tensor(next_available_actions, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                q_next_all = self.local_q_tgt(next_obs.reshape(-1, self.obs_dim)).reshape(B, self.n_agents, self.n_actions)
                if navail is not None:
                    q_next_all = self._mask_q(q_next_all, navail)
                next_act = torch.argmax(q_next_all, dim=-1)

            q_loc_all = self.local_q(obs.reshape(-1, self.obs_dim)).reshape(B, self.n_agents, self.n_actions)
            if avail is not None:
                q_loc_all = self._mask_q(q_loc_all, avail)
            q_loc = torch.gather(q_loc_all, dim=-1, index=act.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                q_loc_next_all = self.local_q_tgt(next_obs.reshape(-1, self.obs_dim)).reshape(B, self.n_agents, self.n_actions)
                if navail is not None:
                    q_loc_next_all = self._mask_q(q_loc_next_all, navail)
                q_loc_next = torch.gather(q_loc_next_all, dim=-1, index=next_act.unsqueeze(-1)).squeeze(-1)

            x = self._build_x(obs, act)
            x_next = self._build_x(next_obs, next_act)

            # IPC uses softmax policy over masked logits
            pi_for_ipc = F.softmax(q_loc_all, dim=-1)
        else:
            act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                na = self._scale_action(self.actor_tgt(next_obs.reshape(-1, self.obs_dim))).reshape(B, self.n_agents, self.act_dim)
                q_loc_next = self.local_critic_tgt(
                    next_obs.reshape(-1, self.obs_dim),
                    na.reshape(-1, self.act_dim),
                ).reshape(B, self.n_agents)

            q_loc = self.local_critic(
                obs.reshape(-1, self.obs_dim),
                act.reshape(-1, self.act_dim),
            ).reshape(B, self.n_agents)

            x = self._build_x(obs, act)
            x_next = self._build_x(next_obs, na)
            pi_for_ipc = None

        # ----- group TD losses -----
        grp_losses, reg_losses, mu_by_group = [], [], []
        for i in range(self.n_groups):
            mask = self._group_mask(i, B)
            lam = self.attn[i](x, mask)
            mu = (lam.unsqueeze(-1) * x).sum(dim=1)
            mu_by_group.append(mu)

            q_gmf = self.gmf_q[i](mu)
            q_loc_grp = (lam * q_loc).sum(dim=1)
            q_grp = q_loc_grp + 2.0 * q_gmf

            with torch.no_grad():
                lam_next = self.attn[i](x_next, mask)
                mu_next = (lam_next.unsqueeze(-1) * x_next).sum(dim=1)
                q_gmf_next = self.gmf_q_tgt[i](mu_next)
                q_loc_grp_next = (lam_next * q_loc_next).sum(dim=1)

                y_i = r_global + self.gamma * (1.0 - done) * (q_loc_grp_next + 2.0 * q_gmf_next)

            grp_losses.append(F.mse_loss(q_grp, y_i))

            if self.reg_alpha > 0:
                diff2 = ((x - mu.unsqueeze(1)) ** 2).sum(dim=-1)
                reg = (diff2 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                reg_losses.append(reg.mean())

        loss_grp = torch.stack(grp_losses).mean()

        # ----- top TD loss -----
        q_tot = 0.0
        for i in range(self.n_groups):
            q_tot = q_tot + self.gmf_q[i](mu_by_group[i])

        with torch.no_grad():
            q_tot_next = 0.0
            for i in range(self.n_groups):
                mask = self._group_mask(i, B)
                lam_next = self.attn[i](x_next, mask)
                mu_next = (lam_next.unsqueeze(-1) * x_next).sum(dim=1)
                q_tot_next = q_tot_next + self.gmf_q_tgt[i](mu_next)

            y = r_global + self.gamma * (1.0 - done) * q_tot_next

        loss_tot = F.mse_loss(q_tot, y)

        # ----- IPC loss (discrete only) -----
        loss_ipc = torch.tensor(0.0, device=self.device)
        if self.ipc_beta > 0 and self.is_discrete:
            for i in range(self.n_groups):
                mask = self._group_mask(i, B)
                pi_bar = self.mean_policy[i].view(1, 1, -1).expand(B, self.n_agents, self.n_actions)
                kl = (pi_for_ipc * (torch.log(pi_for_ipc.clamp(min=1e-8)) - torch.log(pi_bar.clamp(min=1e-8)))).sum(dim=-1)
                loss_ipc = loss_ipc + (kl * mask).sum() / mask.sum().clamp(min=1.0)
            loss_ipc = self.ipc_beta * loss_ipc / self.n_groups

        loss_reg = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=self.device)

        # critic / q update
        loss_q = loss_grp + loss_tot + loss_reg + loss_ipc
        self.opt.zero_grad()
        loss_q.backward()
        self.opt.step()

        # update mean policy
        if update_mean_policy and self.ipc_beta > 0 and self.is_discrete:
            with torch.no_grad():
                for i in range(self.n_groups):
                    mask = self._group_mask(i, B)
                    denom = mask.sum().clamp(min=1.0)
                    pi_i = (pi_for_ipc * mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom
                    self.mean_policy[i] = 0.99 * self.mean_policy[i] + 0.01 * pi_i
                    self.mean_policy[i] = self.mean_policy[i] / self.mean_policy[i].sum().clamp(min=1e-8)

        # actor update (continuous)
        loss_actor = torch.tensor(0.0, device=self.device)
        if not self.is_discrete:
            cur_a = self._scale_action(self.actor(obs.reshape(-1, self.obs_dim))).reshape(B, self.n_agents, self.act_dim)
            x_pi = self._build_x(obs, cur_a)
            q_tot_pi = 0.0
            for i in range(self.n_groups):
                mask = self._group_mask(i, B)
                lam = self.attn[i](x_pi, mask)
                mu = (lam.unsqueeze(-1) * x_pi).sum(dim=1)
                q_tot_pi = q_tot_pi + self.gmf_q[i](mu)
            loss_actor = -q_tot_pi.mean()
            self.actor_opt.zero_grad()
            loss_actor.backward()
            self.actor_opt.step()

            self._soft_update(self.actor_tgt, self.actor, self.tau)
            self._soft_update(self.local_critic_tgt, self.local_critic, self.tau)
        else:
            self._soft_update(self.local_q_tgt, self.local_q, self.tau)

        for i in range(self.n_groups):
            self._soft_update(self.gmf_q_tgt[i], self.gmf_q[i], self.tau)

        return {
            "loss_q": float(loss_q.detach().cpu().item()),
            "loss_grp": float(loss_grp.detach().cpu().item()),
            "loss_tot": float(loss_tot.detach().cpu().item()),
            "loss_reg": float(loss_reg.detach().cpu().item()),
            "loss_ipc": float(loss_ipc.detach().cpu().item()),
            "loss_actor": float(loss_actor.detach().cpu().item()),
        }
