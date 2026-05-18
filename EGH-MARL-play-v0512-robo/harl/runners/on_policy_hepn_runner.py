"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
import copy
import time
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.common.buffers.on_policy_hepn_critic_buffer_ep import OnPolicyHepnCriticBufferEP
from harl.common.buffers.hepn_graph_buffer import GraphBuffer


class OnPolicyHepnRunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""
    def __init__(self, args, algo_args, env_args):
        super(OnPolicyHepnRunner, self).__init__(args, algo_args, env_args)
        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.graph_buffer = GraphBuffer({**algo_args["train"], **algo_args["model"], **algo_args["algo"]})

            share_observation_space = self.envs.share_observation_space[0]
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                self.critic_buffer = OnPolicyHepnCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    use_history=self.use_history,
                    windows_size=self.windows_size
                )

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
            decay_mode = self.algo_args["train"]["lr_decay_mode"]
            if decay_mode is not None:  #decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes, decay_mode)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes, decay_mode)
                self.critic.lr_decay(episode, episodes, decay_mode)

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
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    layer_data,
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

                data = (
                    obs,
                    share_obs,
                    layer_data,
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

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, layer_data, available_actions = self.envs.reset()

        # replay buffer
        self.graph_buffer.graphs[0] = copy.deepcopy(layer_data)
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
        if self.state_type == "EP":
            if self.algo_args["train"]["n_rollout_threads"] > 1:
                share_obs = share_obs.squeeze()
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
            self.critic_buffer.graphs[0] = copy.deepcopy(layer_data)
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()
            self.critic_buffer.graphs[0] = copy.deepcopy(layer_data)

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        # if self.flag:
        obs_list = []
        rnn_states_list = []
        masks_list = []
        for agent_id in range(self.num_agents):
            obs_list.append(self.actor_buffer[agent_id].obs[step])
            rnn_states_list.append(self.actor_buffer[agent_id].rnn_states[step])
            masks_list.append(self.actor_buffer[agent_id].masks[step])
        actions, action_log_probs, rnn_state = self.actor[0].get_actions(
            np.stack(obs_list, axis=0).transpose(1, 0, 2),
            np.stack(rnn_states_list, axis=0), # RNN状态处理有问题
            np.stack(masks_list, axis=0),
            self.graph_buffer.graphs[step],
            None,   # available_action暂时不考虑
        )
        actions = _t2n(actions)
        action_log_probs = _t2n(action_log_probs)
        rnn_states = _t2n(rnn_state).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
                self.critic_buffer.graphs[step]
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
                self.critic_buffer.graphs[step]
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

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            layer_data,
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
        
        self.graph_buffer.insert(layer_data)

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                layer_data,
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, layer_data, rnn_states_critic, values, rewards, masks, bad_masks
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
                self.critic_buffer.graphs[-1]
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
                self.critic_buffer.graphs[-1]
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

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
                self.actor_buffer, self.graph_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        # else:
        #     for agent_id in range(self.num_agents):
        #         if self.state_type == "EP":
        #             actor_train_info = self.actor[agent_id].train(
        #                 self.actor_buffer[agent_id], advantages.copy(), "EP"
        #             )
        #         elif self.state_type == "FP":
        #             actor_train_info = self.actor[agent_id].train(
        #                 self.actor_buffer[agent_id],
        #                 advantages[:, :, agent_id].copy(),
        #                 "FP",
        #             )
        #         actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
    
    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.graph_buffer.after_update()
        self.critic_buffer.after_update()

    # 用于评估模型，绘制动图
    @torch.no_grad()
    def mod_render(self, episodes):
        """Evaluate the model."""
        info_list = []
        pooling_plan = []
        h_edge_index = []
        edge_weights = []
        eval_times = []
        dis = []
        rewards = 0
        done_rewards = []

        self.actor[0].actor.n_threads = 1

        # used for transfor experiments
        # self.actor[0].actor.n_nodes = self.env_args["nr_agents"]
        # self.actor[0].actor.args['nr_agents'] = self.env_args["nr_agents"]
        
        self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
        self.actor[0].actor.local_tool.update_edges()
            # used for transfor experiments
            # self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
        eval_episode = 0
        eval_obs, eval_share_obs, layer_data, eval_available_actions = self.show_envs.reset()

        eval_rnn_states = np.zeros(
            (
                1,
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (1, self.num_agents, 1),
            dtype=np.float32,
        )
        
        while True:
            eval_obs_list = []
            eval_rnn_states_list = []
            eval_masks_list = []
            eval_available_actions_list = []
            for agent_id in range(self.num_agents):
                eval_obs_list.append(eval_obs[:, agent_id])
                eval_rnn_states_list.append(eval_rnn_states[:, agent_id])
                eval_masks_list.append(eval_masks[:, agent_id])
                if eval_available_actions[0] is not None:
                    eval_available_actions_list.append(eval_available_actions[:, agent_id])
            start_time = time.time()
            # start_time = time.perf_counter()
            eval_actions, temp_rnn_state = self.actor[0].act(
                np.stack(eval_obs_list, axis=0).transpose(1, 0, 2),
                np.stack(eval_rnn_states_list, axis=0),
                np.stack(eval_masks_list, axis=0),
                layer_data,
                np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                if len(eval_available_actions_list) > 0
                else None, 
                deterministic=True,
            )
            end_time = time.time()
            # end_time = time.perf_counter()
            eval_time = (end_time - start_time) * 1000
            eval_times.append(eval_time)
            plan = layer_data[0]['interLayer_edgeMat'][1].detach().cpu().numpy()
            sort_idx = np.argsort(plan[1])
            plan = plan[:, sort_idx]
            pooling_plan.append(plan[0].copy())
            edge_index = process_layer_edgeIndex(layer_data, 1)
            h_edge_index.append(edge_index.copy())
            edge_weight = process_layer_edgeWeight(layer_data, 0)
            edge_weights.append(edge_weight.copy())
            eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
            eval_actions = _t2n(eval_actions)


            (
                eval_obs,
                eval_share_obs,
                layer_data,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.show_envs.step(eval_actions)
            
            if self.args["env"] in ['rendezvous', 'pursuit', 'navigation']:
                dis.append(self.show_envs.envs[0].env.world.distance_matrix)
                
            info_list.append(copy.deepcopy(eval_infos[0][0]))
            rewards += eval_rewards.mean()

            eval_dones_env = np.all(eval_dones, axis=1)
            # 一个episode结束
            if eval_dones_env[0]:
                done_rewards.append(rewards)
                rewards = 0
                eval_episode += 1
                self.show_envs.envs[0].make_ani((info_list, np.array(done_rewards), dis, pooling_plan))
                # info_list = []
                # pooling_plan = []
                eval_obs, eval_share_obs, layer_data, eval_available_actions = self.show_envs.reset()
            
            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (1, self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )
            
            if eval_episode >= episodes:
                # self.show_envs.envs[0].env.save_replay()
                self.show_envs.close()
                return info_list, np.array(done_rewards), dis, pooling_plan, h_edge_index, edge_weights, eval_times


def process_layer_edgeIndex(layer_data, layer=0):
    edge_mat_list = []
    start_idx = [0]
    for i, data in enumerate(layer_data):
        mat = copy.deepcopy(data)
        start_idx.append(start_idx[i] + mat['node_size'][layer])
        mat['layer_edgeMat'][layer][0, :] += start_idx[i]
        mat['layer_edgeMat'][layer][1, :] += start_idx[i]
        edge_mat_list.append(mat['layer_edgeMat'][layer])
    edge_mat_list = torch.cat(edge_mat_list, 1)
    return edge_mat_list.detach().cpu().numpy()


def process_layer_edgeWeight(layer_data, layer=0):
    edge_weight_list = []
    for data in layer_data:
        edge_weight_list.append(data['layer_edgeWeight'][layer])
    edge_weight_list = torch.cat(edge_weight_list, 0)
    return edge_weight_list.detach().cpu().numpy()