"""Runner for off-policy HMF (Hierarchical Mean-Field) algorithm."""
import numpy as np
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
from harl.algorithms.hmf_system import HMFAgentSystem
from harl.algorithms.critics.hmf_system_wrapper import HMFSystemWrapperCritic
from harl.utils.trans_tools import _t2n

class OffPolicyHMFRunner(OffPolicyBaseRunner):
    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        obs_dim = self.envs.observation_space[0].shape[0]
        act_space = self.envs.action_space[0]
        n_groups = algo_args["algo"].get("n_groups", 4)

        self.hmf = HMFAgentSystem(
            n_agents=self.num_agents,
            obs_dim=obs_dim,
            act_space=act_space,
            n_groups=n_groups,
            device=str(self.device),
            gamma=algo_args["algo"]["gamma"],
            lr=algo_args["model"]["lr"],
            tau=algo_args["algo"]["polyak"],
            reg_alpha=algo_args["algo"].get("reg_alpha", 0.0),
            ipc_beta=algo_args["algo"].get("ipc_beta", 0.0),
            seed=algo_args["seed"]["seed"],
            hidden=algo_args["model"]["hidden"],
            attn_hidden=algo_args["model"]["attn_hidden"],
            # actor_lr=algo_args["algo"].get("actor_lr", None),
        )

        if act_space.__class__.__name__ == "Discrete":
            self.actor[0].actor = self.hmf.local_q
            self.actor[0].target_actor = self.hmf.local_q_tgt
        else:
            self.actor[0].actor = self.hmf.actor
            self.actor[0].target_actor = self.hmf.actor_tgt

        self.critic = HMFSystemWrapperCritic(self.hmf, self.device)

    # ---- override rollout action selection to pass available_actions even when flag=True ----
    def get_actions(self, obs, available_actions=None, add_random=True):
        act_space = self.envs.action_space[0]
        if act_space.__class__.__name__ == "Discrete" and available_actions is not None and len(np.array(available_actions).shape) == 3:
            n_threads = obs.shape[0]
            obs_cat = np.concatenate(obs)  # (T*N, obs_dim)
            ava_cat = np.concatenate(available_actions)  # (T*N, A)
            actions = self.actor[0].get_actions(obs_cat, ava_cat, add_random)
            actions = _t2n(actions)
            actions = np.array(np.split(actions, n_threads))
            return actions
        # fallback to base implementation
        return super().get_actions(obs, available_actions=available_actions, add_random=add_random)

    def train(self):
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

        obs = np.transpose(sp_obs, (1, 0, 2))
        next_obs = np.transpose(sp_next_obs, (1, 0, 2))

        act_space = self.envs.action_space[0]
        if act_space.__class__.__name__ == "Discrete":
            act = np.transpose(sp_actions, (1, 0, 2)).squeeze(-1).astype(np.int64)
        else:
            act = np.transpose(sp_actions, (1, 0, 2)).astype(np.float32)

        if sp_reward.ndim == 2 and sp_reward.shape[1] == 1:
            rew = np.repeat(sp_reward, self.num_agents, axis=1).astype(np.float32)
        else:
            B = obs.shape[0]
            rew = sp_reward.reshape(B, self.num_agents).astype(np.float32)

        done = sp_done.astype(np.float32)
        info = self.hmf.update(obs, act, rew, next_obs, done, update_mean_policy=True)
        return {}, info
