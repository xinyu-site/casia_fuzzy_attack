"""Runner for off-policy MA algorithms"""
import copy
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
import numpy as np
from harl.utils.trans_tools import _t2n

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
from harl.train_attack.MADDPG import MADDPG

import yaml

class OnPolicyMATrainAttackRunner(OnPolicyBaseRunner):

    def __init__(self, args, algo_args, env_args , model_path):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        
        self.model_path = model_path

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
        new_logger_path = algo_args["logger"]["log_dir"] + '_eval'
        if not self.algo_args["render"]["use_render"] and algo_args['train']['train_flag']:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=new_logger_path,
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

        

    #@torch.no_grad()
    def eval(self,episodes,attack_config, noise_level=0.1,noise_num=4.0):
        """Evaluate the model."""
        print("Evaluate the model.")   
        reward_episode_list = []  
        path = self.model_path
        if self.flag:
            load_name = path + 'actor_agent' + str(0) + '.pt'
            self.actor[0].actor = torch.load(load_name,weights_only=False)
            self.actor[0].actor.n_threads = 1
            #self.actor[0].actor.update_edges()
        else:
            for agent_id in range(self.num_agents):
                load_name = path + 'actor_agent' + str(agent_id) + '.pt'
                self.actor[agent_id].actor = torch.load(load_name)
        value_normalizer_state_dict = torch.load(path+'value_normalizer.pt',weights_only=False)
        self.value_normalizer.load_state_dict(value_normalizer_state_dict)
        #print(f'type of critic: {type(self.critic)}')
        critic_state_dict = torch.load(path+'critic_agent.pt',weights_only=False)
        self.critic.critic.load_state_dict(critic_state_dict)
        
          # logger callback at the beginning of evaluation
        eval_episode = 0
        self.logger.episode_init(eval_episode)
        self.logger.eval_init()
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        #print(eval_available_actions) #None

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

        total_rewards = 0.0

        # prepare the attack model
        # agent_num = self.num_agents
        agent_num = self.num_agents
        obs_dim = self.envs.observation_space[0].shape[0]

        dim_info = {}
        for agent_id in range(self.num_agents):
            dim_info[f'agent_{agent_id}'] = []  # [obs_dim, act_dim]
            dim_info[f'agent_{agent_id}'].append(obs_dim)
            dim_info[f'agent_{agent_id}'].append(obs_dim)
        
        maddpg = MADDPG(dim_info, attack_config["buffer_capacity"], attack_config["batch_size"], attack_config["actor_lr"], attack_config["critic_lr"])
        #print(f'obs_dim: {obs_dim}')

        eval_step = 0
        while True:
            eval_step += 1
            self.actor[0].actor.zero_grad()
            maddpg_actions_list = []
            maddpg_obs_list = []
            maddpg_next_obs_list = []
            maddpg_reward_list = []
            maddpg_done_list = []
            attack_obs = torch.zeros((self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, obs_dim), dtype=torch.float32)
            for thread in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                obsforattack = {f'agent_{agent_id}': eval_obs[thread, agent_id] for agent_id in range(self.num_agents)}
                maddpg_obs_list.append(obsforattack)
                if eval_step > attack_config["ramdom_step"]:
                    attack_actions = maddpg.select_action(obsforattack)
                    #print('ok')
                    maddpg_actions_list.append(attack_actions)
                    for agent_id in range(self.num_agents):
                        attack_obs[thread, agent_id, :] = attack_actions[f'agent_{agent_id}']
                else:
                    attack_obs[thread, :, :] = torch.randn(self.num_agents, 34).clamp(-1, 1)
                    maddpg_actions_list.append({f'agent_{agent_id}': attack_obs[thread, agent_id, :].numpy() for agent_id in range(self.num_agents)})

            #eval_obs = eval_obs + noise_level * attack_obs.numpy()
            attack_obs_np = attack_obs.numpy()
            #print(attack_obs_np.shape)
            # 计算最后一个维度的范数，形状为 (10, 10, 1)
            norms = np.linalg.norm(attack_obs_np, axis=2, keepdims=True)
            # 避免除零，将零范数设置为1
            norms[norms == 0] = 1
            # 对每个34维向量进行归一化，使其范数为noise_num
            attack_obs_normalized = attack_obs_np / norms * noise_num
            #eval_obs = eval_obs + noise_level * attack_obs_normalized
            eval_obs[:,:,4:34] = eval_obs[:,:,4:34] + noise_level * attack_obs_normalized[:,:,4:34]
            eval_obs = np.clip(eval_obs, -1.0, 1.0)  # clip the observation to a reasonable range
            #print(attack_obs[0][0])
            #self.actor[0].actor.zero_grad()
            self.logger.episode_init(
                eval_episode
            )  # logger callback at the beginning of each evaluation episode
            #eval_actions_collector = []
            
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

            self.actor[0].actor.zero_grad()
            eval_actions = eval_actions.reshape(self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 2)
            eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
            eval_actions = _t2n(eval_actions)
                            
            #print(f'eval_actions shape: {eval_actions.shape}')
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            #print(eval_obs[0][0])
            for thread in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                obsforattack = {f'agent_{agent_id}': eval_obs[thread, agent_id] for agent_id in range(self.num_agents)}
                maddpg_next_obs_list.append(obsforattack)
                #maddpg_reward_list.append(-eval_rewards[thread][0])
                maddpg_reward_list.append( {f'agent_{agent_id}': -eval_rewards[thread, agent_id] for agent_id in range(self.num_agents)} )
                maddpg_done_list.append( {f'agent_{agent_id}': eval_dones[thread,agent_id] for agent_id in range(self.num_agents)} )
            #print(maddpg_done_list)
            for thread in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                maddpg.add(maddpg_obs_list[thread], maddpg_actions_list[thread], maddpg_reward_list[thread], maddpg_next_obs_list[thread], maddpg_done_list[thread])

            if eval_step >= attack_config["ramdom_step"] and eval_step % attack_config["learn_interval"] == 0:  # learn every few steps
                maddpg.learn(attack_config["batch_size"],attack_config["gamma"])
                maddpg.update_target(attack_config["tau"])

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(eval_data) # logger callback at each step of evaluation                        
            
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
                    eval_step = 0
                    eval_rewards = np.sum(self.logger.one_episode_rewards[eval_i], axis=0)
                    #print('-')
                    print(f'episode {eval_episode} reward: {eval_rewards[0][0]}')
                    #print(attack_obs[0][0])
                    #print(torch.norm(attack_obs[0][0]))
                    #print(np.linalg.norm(attack_obs_normalized[0][0]))
                    reward_episode_list.append(eval_rewards[0][0])
                    total_rewards += eval_rewards[0][0]
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done
                    #重置环境：
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

            if eval_episode >= episodes:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break
        return total_rewards / episodes , reward_episode_list