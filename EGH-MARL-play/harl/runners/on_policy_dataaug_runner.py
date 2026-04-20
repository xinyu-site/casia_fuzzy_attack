"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
import math
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.utils.trans_tools import _t2n
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY

class OnPolicyDARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA data augmentation algorithms."""
    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        
        # 初始化额外的 actor_buffer 用于数据增强的轨迹
        self.actor_buffer_90 = []
        self.actor_buffer_180 = []
        self.actor_buffer_270 = []

        if self.algo_args["render"]["use_render"] is False:  # train, not render
            for agent_id in range(self.num_agents):
                ac_bu_90 = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                ac_bu_180 = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                ac_bu_270 = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer_90.append(ac_bu_90)
                self.actor_buffer_180.append(ac_bu_180)
                self.actor_buffer_270.append(ac_bu_270)

            share_observation_space = self.envs.share_observation_space[0]
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                self.critic_buffer_90 = OnPolicyCriticBufferEP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
            )
                self.critic_buffer_180 = OnPolicyCriticBufferEP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
            )
                self.critic_buffer_270 = OnPolicyCriticBufferEP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
            )
            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.
                self.critic_buffer_90 = OnPolicyCriticBufferFP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )
                self.critic_buffer_180 = OnPolicyCriticBufferFP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )
                self.critic_buffer_270 = OnPolicyCriticBufferFP(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                self.num_agents,
            )
            else:
                raise NotImplementedError

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode
            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    action_log_probs,
                    # action_log_probs_90,
                    # action_log_probs_180,
                    # action_log_probs_270,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                if self.args["env"] == "pettingzoo_mpe":
                    # expand actions for simple spread
                    new_actions = np.zeros((actions.shape[0], actions.shape[1], 5))
                    new_actions[:, :, 1] = np.where(actions[:, :, 0] > 0, actions[:, :, 0], 0)
                    new_actions[:, :, 2] = np.where(actions[:, :, 0] < 0, -actions[:, :, 0], 0)
                    new_actions[:, :, 3] = np.where(actions[:, :, 1] > 0, actions[:, :, 1], 0)
                    new_actions[:, :, 4] = np.where(actions[:, :, 1] < 0, -actions[:, :, 1], 0)
                    (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                    ) = self.envs.step(new_actions)
                else:
                    (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                    ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.logger.per_step(data)  # logger callback at each step

                self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            actor_train_infos, critic_train_info = self.train()

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, self.actor_buffer_90, self.actor_buffer_180, self.actor_buffer_270, advantages.copy(), self.num_agents, self.state_type
            )
            
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], self.actor_buffer_90[agent_id],self.actor_buffer_180[agent_id],self.actor_buffer_270[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )
                actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.critic_buffer_90, self.critic_buffer_180, self.critic_buffer_270, self.value_normalizer)
        
        return actor_train_infos, critic_train_info
    
    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        share_obs_90 = []
        share_obs_180 = []
        share_obs_270 = []
        temp_share_obs = share_obs[:, 0].copy().reshape(share_obs.shape[0], self.num_agents, -1)
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            
            obs_90 = self.rotation_obs(obs[:, agent_id].copy(), np.pi/2)
            obs_180 = self.rotation_obs(obs[:, agent_id].copy(), np.pi)
            obs_270 = self.rotation_obs(obs[:, agent_id].copy(), 3*np.pi/2)

            temp_share_obs_90 = self.rotation_obs(temp_share_obs[:, agent_id].copy(), np.pi/2)
            share_obs_90.append(temp_share_obs_90)
            temp_share_obs_180 = self.rotation_obs(temp_share_obs[:, agent_id].copy(), np.pi)
            share_obs_180.append(temp_share_obs_180)
            temp_share_obs_270 = self.rotation_obs(temp_share_obs[:, agent_id].copy(), 3*np.pi/2)
            share_obs_270.append(temp_share_obs_270)
            # 
            # share_obs_180.append(obs_180)
            # 
            # share_obs_270.append(obs_270)

            self.actor_buffer_90[agent_id].obs[0] = obs_90
            self.actor_buffer_180[agent_id].obs[0] = obs_180
            self.actor_buffer_270[agent_id].obs[0] = obs_270
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
                self.actor_buffer_90[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
                self.actor_buffer_180[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
                self.actor_buffer_270[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

        share_obs_90 = np.concatenate(share_obs_90, axis=1) #拼接
        share_obs_180 = np.concatenate(share_obs_180, axis=1)
        share_obs_270 = np.concatenate(share_obs_270, axis=1)

        share_obs_90_1 = np.repeat(share_obs_90[:, np.newaxis, :], self.algo_args["train"]["n_rollout_threads"], axis=1)
        share_obs_180_1 = np.repeat(share_obs_180[:, np.newaxis, :], self.algo_args["train"]["n_rollout_threads"], axis=1)
        share_obs_270_1 = np.repeat(share_obs_270[:, np.newaxis, :], self.algo_args["train"]["n_rollout_threads"], axis=1)

        if self.state_type == "EP":
            if self.algo_args["train"]["n_rollout_threads"] > 1:
                share_obs = share_obs.squeeze()
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
            self.critic_buffer_90.share_obs[0] = share_obs_90_1[:, 0].copy()
            self.critic_buffer_180.share_obs[0] = share_obs_180_1[:, 0].copy()
            self.critic_buffer_270.share_obs[0] = share_obs_270_1[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()
            self.critic_buffer_90.share_obs[0] = share_obs[:, 0].copy()
            self.critic_buffer_180.share_obs[0] = share_obs[:, 0].copy()
            self.critic_buffer_270.share_obs[0] = share_obs[:, 0].copy()

    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        if self.flag:
            obs_list = []
            rnn_states_list = []
            masks_list = []
            available_actions_list = []
            for agent_id in range(self.num_agents):
                obs_list.append(self.actor_buffer[agent_id].obs[step])

                rnn_states_list.append(self.actor_buffer[agent_id].rnn_states[step])
                masks_list.append(self.actor_buffer[agent_id].masks[step])
                if self.actor_buffer[agent_id].available_actions is not None:
                    available_actions_list.append(self.actor_buffer[agent_id].available_actions[step])
                
            actions, action_log_probs, rnn_state = self.actor[0].get_actions(
                np.stack(obs_list, axis=0).transpose(1, 0, 2),
                np.stack(rnn_states_list, axis=0), # RNN状态处理有问题
                np.stack(masks_list, axis=0),
                np.stack(available_actions_list, axis=0).transpose(1, 0, 2) if len(available_actions_list) > 0
                else None,
            )

            actions = _t2n(actions)
            action_log_probs = _t2n(action_log_probs)
            rnn_states = _t2n(rnn_state).transpose(1, 0, 2, 3)

        else:
            action_collector = []
            action_log_prob_collector = []
            rnn_state_collector = []
            for agent_id in range(self.num_agents):
                action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                    self.actor_buffer[agent_id].obs[step],
                    self.actor_buffer[agent_id].rnn_states[step],
                    self.actor_buffer[agent_id].masks[step],
                    self.actor_buffer[agent_id].available_actions[step]
                    if self.actor_buffer[agent_id].available_actions is not None
                    else None,
                )
                action_collector.append(_t2n(action))
                action_log_prob_collector.append(_t2n(action_log_prob))
                rnn_state_collector.append(_t2n(rnn_state))
            # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
            actions = np.array(action_collector).transpose(1, 0, 2)
            action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
            rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    # , action_log_probs_90, action_log_probs_180, action_log_probs_270, 


    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )
        temp_share_obs = share_obs[:, 0].copy().reshape(share_obs.shape[0], self.num_agents, -1)
        share_obs_90 = []
        share_obs_180 = []
        share_obs_270 = []
        for agent_id in range(self.num_agents):

            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

            if self.args["env"] == 'mujoco3d':
                actions_90 = actions[:, agent_id].copy()
                actions_180 = actions[:, agent_id].copy()
                actions_270 = actions[:, agent_id].copy()
            elif self.args["env"] == 'smacv2':
                actions_90 = actions[:, agent_id].copy()
                actions_90[actions[:, agent_id] == 2] = 5  # 北 -> 西
                actions_90[actions[:, agent_id] == 5] = 3  # 西 -> 南
                actions_90[actions[:, agent_id] == 3] = 4  # 南 -> 东
                actions_90[actions[:, agent_id] == 4] = 2  # 东 -> 北

                actions_180 = actions[:, agent_id].copy()
                actions_180[actions[:, agent_id] == 2] = 3  # 北 -> 南
                actions_180[actions[:, agent_id] == 5] = 4  # 西 -> 东
                actions_180[actions[:, agent_id] == 3] = 2  # 南 -> 北
                actions_180[actions[:, agent_id] == 4] = 5  # 东 -> 西

                actions_270 = actions[:, agent_id].copy()
                actions_270[actions[:, agent_id] == 2] = 4  # 北 -> 东
                actions_270[actions[:, agent_id] == 5] = 2  # 西 -> 北
                actions_270[actions[:, agent_id] == 3] = 5  # 南 -> 西
                actions_270[actions[:, agent_id] == 4] = 3  # 东 -> 南
            else:
                actions_90 = self.rotation_action(actions[:, agent_id], np.pi/2)
                actions_180 = self.rotation_action(actions[:, agent_id], np.pi)
                actions_270 = self.rotation_action(actions[:, agent_id], 3*np.pi/2)


            obs_90 = self.rotation_obs(obs[:, agent_id], np.pi/2)
            temp_share_obs_90 = self.rotation_obs(temp_share_obs[:, agent_id], np.pi / 2)
            share_obs_90.append(temp_share_obs_90)
            # share_obs_90.append(obs_90)
            self.actor_buffer_90[agent_id].insert(
                obs_90,
                rnn_states[:, agent_id],
                actions_90,
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )
            
            obs_180 = self.rotation_obs(obs[:, agent_id],np.pi)
            temp_share_obs_180 = self.rotation_obs(temp_share_obs[:, agent_id], np.pi / 2)
            share_obs_180.append(temp_share_obs_180)
            # share_obs_180.append(obs_180)
            self.actor_buffer_180[agent_id].insert(
                obs_180,
                rnn_states[:, agent_id],
                actions_180,
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

            obs_270 = self.rotation_obs(obs[:, agent_id],3*np.pi/2)
            temp_share_obs_270 = self.rotation_obs(temp_share_obs[:, agent_id], 3*np.pi / 2)
            share_obs_270.append(temp_share_obs_270)
            # share_obs_270.append(obs_270)
            self.actor_buffer_270[agent_id].insert(
                obs_270,
                rnn_states[:, agent_id],
                actions_270,
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )
        share_obs_90=np.concatenate(share_obs_90, axis=1) #拼接
        share_obs_180=np.concatenate(share_obs_180, axis=1)
        share_obs_270=np.concatenate(share_obs_270, axis=1)

        share_obs_90_1 = np.repeat(share_obs_90[:, np.newaxis, :], 10, axis=1)
        share_obs_180_1 = np.repeat(share_obs_180[:, np.newaxis, :], 10, axis=1)
        share_obs_270_1 = np.repeat(share_obs_270[:, np.newaxis, :], 10, axis=1)

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
            self.critic_buffer_90.insert(
                share_obs_90_1[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
            self.critic_buffer_180.insert(
                share_obs_180_1[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
            self.critic_buffer_270.insert(
                share_obs_270_1[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks
            )
            self.critic_buffer_90.insert(
                share_obs_90_1, rnn_states_critic, values, rewards, masks, bad_masks
            )
            self.critic_buffer_180.insert(
                share_obs_180_1, rnn_states_critic, values, rewards, masks, bad_masks
            )
            self.critic_buffer_270.insert(
                share_obs_270_1, rnn_states_critic, values, rewards, masks, bad_masks
            )

    # def rotation_obs(self,obs,angle):
      
    #     # 定义旋转矩阵
    #     rotation_matrices = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    #     last_two_features = obs[:, -2:]
    #     rotated_features = np.dot(last_two_features, rotation_matrices)
    #     obs[:, -2:] = rotated_features

    #     last_two_features = obs[:, [-4,-3]]
    #     rotated_features = np.dot(last_two_features, rotation_matrices)
    #     obs[:,  [-4,-3]] = rotated_features

    #     return obs
    
    # def rotation_action(self,action,angle):
    #     # 定义旋转矩阵
    #     rotation_matrices = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    #     rotated_action=np.dot(action, rotation_matrices)

    #     return rotated_action

    def rotation_obs(self, obs, angle):
        equ_feature = obs[:,  :self.env_args["equ_nf"]]  # [n_num_rollouts, num_equ * dimension]

        # 定义旋转矩阵
        if self.args["env"] == 'mujoco3d':
            rotation_matrices = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            equ_feature = equ_feature.reshape(equ_feature.shape[0], -1, 3).transpose(0, 2, 1)
        else:
            rotation_matrices = np.array([
                [np.cos(angle), -np.sin(angle)], 
                [np.sin(angle), np.cos(angle)]
            ])
            equ_feature = equ_feature.reshape(equ_feature.shape[0], -1, 2).transpose(0, 2, 1)

        # 应用旋转
        # rotated_features = np.empty_like(equ_feature)
        rotated_features = rotation_matrices @ equ_feature
        # for i in range(equ_feature.shape[0]):
        #     pos = equ_feature[i, :2] 
        #     vel = equ_feature[i, 2:]  
        #     # 分别旋转位置和速度
        #     rotated_pos = np.dot(rotation_matrices, pos)
        #     rotated_vel = np.dot(rotation_matrices, vel)
        #     # 重新组合特征
        #     rotated_features[i, :2] = rotated_pos
        #     rotated_features[i, 2:] = rotated_vel

        # 将旋转后的特征拼接回原始数组的相应位置
        rotated_features = rotated_features.transpose(0, 2, 1).reshape(equ_feature.shape[0], -1)
        obs[:, :self.env_args["equ_nf"]] = rotated_features
        return obs
    
    def rotation_action(self, action, angle):
        # 定义旋转矩阵
        rotation_matrices = np.array([
            [np.cos(angle), -np.sin(angle)], 
            [np.sin(angle), np.cos(angle)]
        ])

        # 应用旋转
        rotated_action = np.empty_like(action)
        for i in range(action.shape[0]):
            rotated_action[i] = np.dot(rotation_matrices, action[i])

        return rotated_action
    
    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
            self.actor_buffer_90[agent_id].after_update()
            self.actor_buffer_180[agent_id].after_update()
            self.actor_buffer_270[agent_id].after_update()
        self.critic_buffer.after_update()
        self.critic_buffer_90.after_update()
        self.critic_buffer_180.after_update()
        self.critic_buffer_270.after_update()