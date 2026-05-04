"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import copy
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY
from gymnasium import spaces


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        
        # 使用时序数据
        self.use_history = env_args['use_history']
        self.windows_size = env_args['windows_size']

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.single_actor = algo_args["algo"]["single_actor"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.flag = self.share_param and self.single_actor
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"] and algo_args['train']['train_flag']:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
            
            self.show_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    1,
                    env_args,
                )
            )
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )

        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        env_args["num_agents"] = self.num_agents
        env_args["env_name"] = args["env"]

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)
        
        if self.args["env"] == "pettingzoo_mpe":
            for i in range(len(self.envs.action_space)):
                self.envs.action_space[i] = spaces.Box(low=0, high=1, shape=(2,))

        # actor
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"], **env_args, **algo_args["train"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            if self.single_actor:
                for agent_id in range(1, self.num_agents):
                    assert (
                        self.envs.observation_space[agent_id]
                        == self.envs.observation_space[0]
                    ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                    assert (
                        self.envs.action_space[agent_id] == self.envs.action_space[0]
                    ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
            else:
                for agent_id in range(1, self.num_agents):
                    assert (
                        self.envs.observation_space[agent_id]
                        == self.envs.observation_space[0]
                    ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                    assert (
                        self.envs.action_space[agent_id] == self.envs.action_space[0]
                    ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                    self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"], **env_args, **algo_args["train"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    use_history=self.use_history,
                    windows_size=self.windows_size
                )
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            self.critic = CRITIC_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"], **env_args, **algo_args["train"]},
                share_observation_space,
                device=self.device,
            )
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    use_history=self.use_history,
                    windows_size=self.windows_size
                )
            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                raise NotImplementedError

            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None
            if algo_args['train']['train_flag']:
                self.logger = LOGGER_REGISTRY[args["env"]](
                    args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
                )
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()
        
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
            #print(f'state_type: {self.state_type}')
            
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
                #print(obs.shape)
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

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
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
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
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
            #print(len(obs_list))
            actions, action_log_probs, rnn_state = self.actor[0].get_actions(
                np.stack(obs_list, axis=0).transpose(1, 0, 2),
                np.stack(rnn_states_list, axis=0), # RNN状态处理有问题
                np.stack(masks_list, axis=0),
                np.stack(available_actions_list, axis=0).transpose(1, 0, 2) if len(available_actions_list) > 0
                else None,
            )
            # print(f'actions shape: {actions.shape}')  10 * 10 * 2
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
            #print(f'type of critic: {type(self.critic)}')
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

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
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
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            if self.flag:
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
                eval_actions, temp_rnn_state = self.actor[0].act(
                    np.stack(eval_obs_list, axis=0).transpose(1, 0, 2),
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
                eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
                eval_actions = _t2n(eval_actions)
            else:
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

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
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    if self.flag:
                        eval_obs_list = []
                        eval_rnn_states_list = []
                        eval_masks_list = []
                        eval_available_actions_list = []
                        for agent_id in range(self.num_agents):
                            eval_obs_list.append(eval_obs[:, agent_id])
                            eval_rnn_states_list.append(eval_rnn_states[:, agent_id])
                            eval_masks_list.append(eval_masks[:, agent_id])
                            if eval_available_actions is not None:
                                eval_available_actions_list.append(eval_available_actions[:, agent_id])
                        eval_actions, temp_rnn_state = self.actor[0].act(
                            np.stack(eval_obs_list, axis=0).transpose(1, 0, 2),
                            np.stack(eval_rnn_states_list, axis=0),
                            np.stack(eval_masks_list, axis=0),
                            np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                            if len(eval_available_actions_list) > 0
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
                        eval_actions = _t2n(eval_actions)
                    else:
                        for agent_id in range(self.num_agents):
                            eval_actions, temp_rnn_state = self.actor[agent_id].act(
                                eval_obs[:, agent_id],
                                eval_rnn_states[:, agent_id],
                                eval_masks[:, agent_id],
                                eval_available_actions[:, agent_id]
                                if eval_available_actions is not None
                                else None,
                                deterministic=True,
                            )
                            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                            eval_actions_collector.append(_t2n(eval_actions))
                        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    if self.args["env"] == "pettingzoo_mpe":
                        new_actions = np.zeros((eval_actions.shape[0], eval_actions.shape[1], 5))
                        new_actions[:, :, 1] = np.where(eval_actions[:, :, 0] > 0, eval_actions[:, :, 0], 0)
                        new_actions[:, :, 2] = np.where(eval_actions[:, :, 0] < 0, -eval_actions[:, :, 0], 0)
                        new_actions[:, :, 3] = np.where(eval_actions[:, :, 1] > 0, eval_actions[:, :, 1], 0)
                        new_actions[:, :, 4] = np.where(eval_actions[:, :, 1] < 0, -eval_actions[:, :, 1], 0)
                        (
                            eval_obs,
                            _,
                            eval_rewards,
                            eval_dones,
                            _,
                            eval_available_actions,
                        ) = self.envs.step(new_actions[0])
                    else:
                        (
                            eval_obs,
                            _,
                            eval_rewards,
                            eval_dones,
                            _,
                            eval_available_actions,
                        ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    if self.flag:
                        eval_obs_list = []
                        eval_rnn_states_list = []
                        eval_masks_list = []
                        # eval_available_actions_list = []
                        for agent_id in range(self.num_agents):
                            eval_obs_list.append(eval_obs[:, agent_id])
                            eval_rnn_states_list.append(eval_rnn_states[:, agent_id])
                            eval_masks_list.append(eval_masks[:, agent_id])
                            # if eval_available_actions[0] is not None:
                            #     eval_available_actions_list.append(eval_available_actions[:, agent_id])
                        eval_actions, temp_rnn_state = self.actor[0].act(
                            np.stack(eval_obs_list, axis=0),
                            np.stack(eval_rnn_states_list, axis=0),
                            np.stack(eval_masks_list, axis=0),
                            None,  # available_actions暂时不考虑
                            deterministic=True
                        )
                        eval_rnn_states = _t2n(temp_rnn_state)
                        eval_actions = _t2n(eval_actions).transpose(1, 0, 2)
                    else:
                        for agent_id in range(self.num_agents):
                            eval_actions, temp_rnn_state = self.actor[agent_id].act(
                                eval_obs[:, agent_id],
                                eval_rnn_states[:, agent_id],
                                eval_masks[:, agent_id],
                                eval_available_actions[:, agent_id]
                                if eval_available_actions[0] is not None
                                else None,
                                deterministic=True,
                            )
                            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                            eval_actions_collector.append(_t2n(eval_actions))
                        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def prep_rollout(self):
        """Prepare for rollout."""
        if self.flag:
            # for agent_id in range(self.num_agents):
            self.actor[0].prep_rollout()
        else:
            for agent_id in range(self.num_agents):
                self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        if self.flag:
        #     for _ in range(self.num_agents):
            self.actor[0].prep_training()
        else:
            for agent_id in range(self.num_agents):
                self.actor[agent_id].prep_training()
        self.critic.prep_training()

    def save(self):
        """Save model parameters."""
        if self.flag:
            policy_actor = self.actor[0].actor
            torch.save(
                policy_actor,
                str(self.save_dir) + "/actor_agent" + str(0) + ".pt",
            )
        else:
            for agent_id in range(self.num_agents):
                policy_actor = self.actor[agent_id].actor
                torch.save(
                    policy_actor,
                    str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
                )
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def restore(self):
        """Restore model parameters."""
        if self.flag:
            policy_actor_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(0)
                + ".pt",
            )
            # self.actor[0].actor.load_state_dict(policy_actor_state_dict)
            self.actor[0].actor = policy_actor_state_dict
            # if self.actor[0].actor.__class__.__name__ == 'EghnPolicy':
            #     self.actor[0].actor.n_threads = 1
            #     self.actor[0].actor.update_edges()
            if 'Eg' in self.actor[0].actor.__class__.__name__:
                self.actor[0].actor.n_threads = 1
                self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
                self.actor[0].actor.local_tool.update_edges()
            elif "G" in self.actor[0].actor.__class__.__name__ or "Hama" in self.actor[0].actor.__class__.__name__:
                self.actor[0].actor.n_threads = 1
                self.actor[0].actor.local_tool.update_edges()
        else:
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/actor_agent"
                    + str(agent_id)
                    + ".pt",
                )
                # self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
                self.actor[agent_id].actor = policy_actor_state_dict
        if not self.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"]) + "/critic_agent" + ".pt"
            )
            self.critic.critic.load_state_dict(policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            if self.algo_args['train']['train_flag']:
                self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
                self.writter.close()
                self.logger.close()
    
    # 用于评估模型，绘制动图
    @torch.no_grad()
    def mod_render(self, episodes):
        """Evaluate the model."""
        from PIL import Image, ImageDraw, ImageFont
        import imageio, os

        info_list = []
        pooling_plan = []
        eval_times = []
        h_edge_index = []
        edge_weights = []
        # ratio = []
        dis = []
        rewards = 0
        done_rewards = []
        imgs = []  # 用于绘制mujoco动图

        self.actor[0].actor.n_threads = 1

        # used for transfor experiments
        # self.actor[0].actor.n_nodes = self.env_args["nr_agents"]
        # self.actor[0].actor.args['nr_agents'] = self.env_args["nr_agents"]
        
        if "Eg" in self.actor[0].actor.__class__.__name__:
            self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
            self.actor[0].actor.local_tool.update_edges()
        elif "G" in self.actor[0].actor.__class__.__name__ or "Hama" in self.actor[0].actor.__class__.__name__:
            self.actor[0].actor.local_tool.update_edges()
        
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.show_envs.reset()
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
        
        if self.args["env"] in ['mujoco3d']:
            video_demo_dir = os.path.join("gifs", self.args["env"], self.env_args['scenario'])
            os.makedirs(video_demo_dir, exist_ok=True)
            img = self.show_envs.render(mode='rgb_array')
            image = Image.fromarray(np.rot90(img[0], k=2), "RGB")
            imgs.append(image)
        elif self.args["env"] == 'smacv2':
            video_demo_dir = os.path.join("gifs", self.args["env"], self.env_args['map_name'])
            os.makedirs(video_demo_dir, exist_ok=True)
            img = self.show_envs.render(mode='rgb_array')
            image = Image.fromarray(img[0], "RGB")
            imgs.append(image)
        
        while True:
            eval_actions_collector = []
            if self.flag:
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
                # 增加推理时间计算
                start_time = time.time()
                eval_actions, temp_rnn_state = self.actor[0].act(
                    np.stack(eval_obs_list, axis=0).transpose(1, 0, 2),
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
                end_time = time.time()
                eval_time = end_time - start_time
                eval_times.append(eval_time)
                if "Eghn" in self.actor[0].actor.__class__.__name__:
                    plan = self.actor[0].actor.eghn_model.current_pooling_plan
                    plan_group = plan.argmax(dim=1).detach().cpu().numpy()
                    edge_weight = self.actor[0].actor.edge_weight.detach().cpu().numpy()
                    edge_weight = edge_weight.reshape(edge_weight.shape[0], )
                    edge_index = self.actor[0].actor.h_edge_index.detach().cpu().numpy()
                    pooling_plan.append(plan_group)
                    edge_weights.append(edge_weight)
                    h_edge_index.append(edge_index)

                eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
                eval_actions = _t2n(eval_actions)
            else:
                eval_time = 0
                for agent_id in range(self.num_agents):
                    start_time = time.time()
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    end_time = time.time()
                    eval_time += (end_time - start_time)
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                eval_time = eval_time * 1000
                eval_times.append(eval_time)

                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.show_envs.step(eval_actions)
            
            if self.args["env"] in ['rendezvous', 'pursuit', 'navigation']:
                dis.append(self.show_envs.envs[0].env.world.distance_matrix)
                
            # print(eval_infos[0][0]["evader_states"])
            info_list.append(copy.deepcopy(eval_infos[0][0]))
            rewards += eval_rewards.mean()

            if self.args["env"] in ['mujoco3d']:
                img = self.show_envs.render(mode='rgb_array')
                imge = Image.fromarray(np.rot90(img[0], k=2), "RGB")
                draw = ImageDraw.Draw(imge)
                font = ImageFont.truetype("harl/envs/mujoco3d/misc/sans-serif.ttf", 20)
                draw.text(
                    (100, 10), "Distance: " + str(eval_infos[0][0]['dist']), (255, 255, 0), font=font
                )
                draw.text(
                    (100, 32), "Instant Reward: " + str(eval_rewards.mean()), (255, 255, 0), font=font
                )
                draw.text(
                    (100, 54),
                    "Episode Reward: " + str(rewards),
                    (255, 255, 0),
                    font=font,
                )
                # draw.text(
                #     (100, 76),
                #     "Episode Timesteps: " + str(episode_timesteps_list[i]),
                #     (255, 255, 0),
                #     font=font,
                # )
                imgs.append(imge)
            elif self.args["env"] == 'smacv2':
                img = self.show_envs.render(mode='rgb_array')
                imge = Image.fromarray(img[0], "RGB")
                imgs.append(imge)

            eval_dones_env = np.all(eval_dones, axis=1)
            # 一个episode结束
            if eval_dones_env[0]:
                if self.args['env'] in ["smacv2", "mujoco3d"]:
                    gif_save_path = os.path.join(video_demo_dir, str(eval_episode) + ".gif")
                    imageio.mimsave(gif_save_path, imgs, fps=60)
                    imgs = []
                done_rewards.append(rewards)
                rewards = 0
                eval_episode += 1
            
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


    def rot_obs(self, obs, angle):
        equ_feature = obs[:, :, :self.env_args["equ_nf"]]  # [n_num_rollouts, num_equ * dimension]

        # 定义旋转矩阵
        rotation_matrices = np.array([
            [np.cos(angle), -np.sin(angle)], 
            [np.sin(angle), np.cos(angle)]
        ])
        equ_feature = equ_feature.reshape(equ_feature.shape[0], equ_feature.shape[1], -1, 2).transpose(0, 1, 3, 2)

        # 应用旋转
        # rotated_features = np.empty_like(equ_feature)
        rotated_features = rotation_matrices @ equ_feature

        # 将旋转后的特征拼接回原始数组的相应位置
        rotated_features = rotated_features.transpose(0, 1, 3, 2).reshape(equ_feature.shape[0], equ_feature.shape[1], -1)
        obs[:, :, :self.env_args["equ_nf"]] = rotated_features
        return obs

    def rot_action(self, action, angle):
        # 定义旋转矩阵
        rotation_matrices = np.array([
            [np.cos(angle), -np.sin(angle)], 
            [np.sin(angle), np.cos(angle)]
        ])

        # 应用旋转
        rotated_action = action @ rotation_matrices.T
        # for i in range(action.shape[0]):
            # rotated_action[i] = np.dot(rotation_matrices, action[i])

        return rotated_action

    def make_perm_ccw(self, A, deg):
        perm = np.arange(A)

        if deg == np.pi / 2:
            perm[2], perm[3], perm[4], perm[5] = 4, 5, 3, 2
        elif deg == np.pi:
            perm[2], perm[3], perm[4], perm[5] = 3, 2, 5, 4
        elif deg == 3 * np.pi / 2:
            perm[2], perm[3], perm[4], perm[5] = 5, 4, 2, 3
        else:
            raise ValueError("deg must be one of {90,180,270}")
        return perm

    # 对称性动作检测
    @torch.no_grad()
    def sym_test(self, episodes):
        """Evaluate the model."""

        eval_times = []
        angles = [np.pi / 2, np.pi, 3 * np.pi / 2]
        err_list = [[] for _ in range(len(angles))]
        errs = [[] for _ in range(len(angles))]
        # rot_eval_available_actions_list = [[] for _ in range(len(angles))]
        rewards = 0

        self.actor[0].actor.n_threads = 1

        if "Eg" in self.actor[0].actor.__class__.__name__:
            self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
            self.actor[0].actor.local_tool.update_edges()
        elif "G" in self.actor[0].actor.__class__.__name__:
            self.actor[0].actor.local_tool.update_edges()

        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.show_envs.reset()

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
            eval_actions_collector = []
            if self.flag:
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
                eval_obs_list = np.stack(eval_obs_list, axis=0).transpose(1, 0, 2)
                # 增加推理时间计算
                start_time = time.time()
                eval_actions, temp_rnn_state = self.actor[0].act(
                    eval_obs_list,
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
                end_time = time.time()
                eval_time = end_time - start_time
                eval_times.append(eval_time)

                eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
                eval_actions = _t2n(eval_actions)

                if self.args["env"] == 'smacv2':
                    eval_available_actions_list = np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2)
                    _, _, action_distribution = self.actor[0].evaluate_actions(
                        eval_obs_list.reshape(-1, eval_obs_list.shape[-1]),
                        np.stack(eval_rnn_states_list, axis=0).reshape(-1, eval_rnn_states_list[0].shape[-2], eval_rnn_states_list[0].shape[-1]),
                        eval_actions.reshape(-1, eval_actions.shape[-1]),
                        np.stack(eval_masks_list, axis=0).reshape(-1, eval_masks_list[0].shape[-1]),
                        eval_available_actions_list.reshape(-1, eval_available_actions_list.shape[-1])
                    )
                    p = action_distribution.probs.cpu().numpy()
                    for i, angle in enumerate(angles):
                        perm = self.make_perm_ccw(eval_available_actions_list.shape[-1], angle)
                        rot_eval_available_actions_list = eval_available_actions_list[..., perm]
                        rot_eval_obs = self.rot_obs(eval_obs_list.copy(), angle)
                        _, _, rot_action_distribution = self.actor[0].evaluate_actions(
                            rot_eval_obs.reshape(-1, eval_obs_list.shape[-1]),
                            np.stack(eval_rnn_states_list, axis=0).reshape(-1, eval_rnn_states_list[0].shape[-2], eval_rnn_states_list[0].shape[-1]),
                            eval_actions.reshape(-1, eval_actions.shape[-1]),
                            np.stack(eval_masks_list, axis=0).reshape(-1, eval_masks_list[0].shape[-1]),
                            rot_eval_available_actions_list.reshape(-1, eval_available_actions_list.shape[-1])                        
                        )
                        rot_p = rot_action_distribution.probs.cpu().numpy()
                        real_rot_p = p[..., perm]
                        common = (eval_available_actions_list > 0) & (rot_eval_available_actions_list > 0)
                        p1 = rot_p * common
                        p2 = real_rot_p * common
                        p1 = p1 / (p1.sum(axis=-1, keepdims=True) + 1e-8)
                        p2 = p2 / (p2.sum(axis=-1, keepdims=True) + 1e-8)

                        # TV distance: 0.5 * L1
                        tv = 0.5 * np.sum(np.abs(p1 - p2), axis=-1)
                        err_list[i].append(tv.mean())

                else:
                    for i, angle in enumerate(angles):
                        rot_eval_obs = self.rot_obs(eval_obs_list.copy(), angle)
                        rot_eval_actions, _ = self.actor[0].act(
                            rot_eval_obs,
                            np.stack(eval_rnn_states_list, axis=0),
                            np.stack(eval_masks_list, axis=0),
                            np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                            if len(eval_available_actions_list) > 0
                            else None, 
                            deterministic=True,
                        )
                        rot_eval_actions = _t2n(rot_eval_actions)
                        real_rot_eval_actions = self.rot_action(eval_actions.copy(), angle)

                        # 使动作在环境动作空间范围内
                        clipped_rot_eval_actions = np.clip(rot_eval_actions, self.envs.action_space[0].low, self.envs.action_space[0].high)
                        clipped_real_rot_eval_actions = np.clip(real_rot_eval_actions, self.envs.action_space[0].low, self.envs.action_space[0].high)
                        # 计算环境中的真实动作值
                        # rot_env_actions = clipped_rot_eval_actions * 10
                        # real_rot_env_actions = clipped_real_rot_eval_actions * 10
                        # rot_norm = np.linalg.norm(rot_env_actions, axis=-1)
                        # real_rot_norm = np.linalg.norm(real_rot_env_actions, axis=-1)
                        # rot_scale = 10 / (rot_norm + 1e-8)
                        # real_rot_scale = 10 / (real_rot_norm + 1e-8)
                        # rot_env_actions = np.where(
                        #     (rot_norm > 10)[..., None],
                        #     rot_env_actions * rot_scale[..., None],
                        #     rot_env_actions,
                        # )
                        # real_rot_env_actions = np.where(
                        #     (real_rot_norm > 10)[..., None],
                        #     real_rot_env_actions * real_rot_scale[..., None],
                        #     real_rot_env_actions,
                        # )
                        
                        # err_agent = np.linalg.norm(rot_env_actions - real_rot_env_actions, axis=-1)
                        err_agent = np.linalg.norm(clipped_rot_eval_actions - clipped_real_rot_eval_actions, axis=-1)
                        err = err_agent.mean()
                        err_list[i].append(err)

            else:
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.show_envs.step(eval_actions)
            
            rewards += eval_rewards.mean()

            eval_dones_env = np.all(eval_dones, axis=1)
            # 一个episode结束
            if eval_dones_env[0]:
                rewards = 0
                eval_episode += 1
                for i in range(len(angles)):
                    errs[i].append(np.mean(err_list[i]))
                    err_list[i] = []
            
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
                self.show_envs.close()
                # return info_list, np.array(done_rewards), dis, pooling_plan, h_edge_index, eval_times
                return errs, eval_times



    @torch.no_grad()
    def transfor_exp(self, episodes):
        """Evaluate the model."""
        from PIL import Image, ImageDraw, ImageFont
        import imageio, os

        info_list = []
        pooling_plan = []
        # ratio = []
        dis = []
        rewards = 0
        done_rewards = []
        # imgs = []  # 用于绘制mujoco动图

        self.actor[0].actor.n_threads = 1

        # used for transfor experiments
        self.actor[0].actor.n_nodes = self.env_args["nr_agents"]
        self.actor[0].actor.args['nr_agents'] = self.env_args["nr_agents"]
        
        if "Eg" in self.actor[0].actor.__class__.__name__ or "G" in self.actor[0].actor.__class__.__name__:
            self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
            self.actor[0].actor.local_tool.update_edges()
            # used for transfor experiments
            # self.actor[0].actor.update_local_tool(self.envs.observation_space[0])
        eval_episode = 0
        eval_obs, eval_share_obs, eval_available_actions = self.show_envs.reset()
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
        
        # if self.args["env"] in ['mujoco3d']:
        #     video_demo_dir = os.path.join("gifs", self.args["env"], self.env_args['scenario'])
        #     os.makedirs(video_demo_dir, exist_ok=True)
        #     img = self.show_envs.render(mode='rgb_array')
        #     image = Image.fromarray(np.rot90(img[0], k=2), "RGB")
        #     imgs.append(image)
        # elif self.args["env"] == 'smacv2':
        #     video_demo_dir = os.path.join("gifs", self.args["env"], self.env_args['map_name'])
        #     os.makedirs(video_demo_dir, exist_ok=True)
        #     img = self.show_envs.render(mode='rgb_array')
        #     image = Image.fromarray(img[0], "RGB")
        #     imgs.append(image)
        
        while True:
            eval_actions_collector = []
            if self.flag:
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
                eval_actions, temp_rnn_state = self.actor[0].act(
                    np.stack(eval_obs_list, axis=0).transpose(1, 0, 2),
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
                if "Eghn" in self.actor[0].actor.__class__.__name__:
                    plan = self.actor[0].actor.eghn_model.current_pooling_plan
                    plan_group = plan.argmax(dim=1).detach().cpu().numpy()
                    pooling_plan.append(plan_group)
                eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
                eval_actions = _t2n(eval_actions)
            else:
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.show_envs.step(eval_actions)
            
            if self.args["env"] in ['rendezvous', 'pursuit', 'navigation']:
                dis.append(self.show_envs.envs[0].env.world.distance_matrix)
                
            # print(eval_infos[0][0]["evader_states"])
            info_list.append(copy.deepcopy(eval_infos[0][0]))
            rewards += eval_rewards.mean()

            # if self.args["env"] in ['mujoco3d']:
            #     img = self.show_envs.render(mode='rgb_array')
            #     imge = Image.fromarray(np.rot90(img[0], k=2), "RGB")
            #     draw = ImageDraw.Draw(imge)
            #     font = ImageFont.truetype("harl/envs/mujoco3d/misc/sans-serif.ttf", 20)
            #     draw.text(
            #         (100, 10), "Distance: " + str(eval_infos[0][0]['dist']), (255, 255, 0), font=font
            #     )
            #     draw.text(
            #         (100, 32), "Instant Reward: " + str(eval_rewards.mean()), (255, 255, 0), font=font
            #     )
            #     draw.text(
            #         (100, 54),
            #         "Episode Reward: " + str(rewards),
            #         (255, 255, 0),
            #         font=font,
            #     )
            #     # draw.text(
            #     #     (100, 76),
            #     #     "Episode Timesteps: " + str(episode_timesteps_list[i]),
            #     #     (255, 255, 0),
            #     #     font=font,
            #     # )
            #     imgs.append(imge)
            # elif self.args["env"] == 'smacv2':
            #     img = self.show_envs.render(mode='rgb_array')
            #     imge = Image.fromarray(img[0], "RGB")
            #     imgs.append(imge)

            eval_dones_env = np.all(eval_dones, axis=1)
            # 一个episode结束
            if eval_dones_env[0]:
                # if self.args['env'] in ["smacv2", "mujoco3d"]:
                #     gif_save_path = os.path.join(video_demo_dir, str(eval_episode) + ".gif")
                #     imageio.mimsave(gif_save_path, imgs, fps=60)
                #     imgs = []
                done_rewards.append(rewards)
                rewards = 0
                eval_episode += 1
            
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
                return info_list, np.array(done_rewards), dis, pooling_plan