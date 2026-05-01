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

def add_rotation_noise_all(vel_cmd, theta):
    """
    vel_cmd: [batch, robots, 2]
    theta: 干扰旋转角（弧度），可以是标量（对所有样本相同）
           或 [batch, robots] 每个机器人不同角度
    """
    theta = torch.tensor(theta)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # 分离 vx, vy
    vx = vel_cmd[..., 0]
    vy = vel_cmd[..., 1]

    # 旋转
    vx_rot = vx * cos_t - vy * sin_t
    vy_rot = vx * sin_t + vy * cos_t

    # 组合回 [..., 2]
    vel_noisy = torch.stack([vx_rot, vy_rot], dim=-1)
    return vel_noisy

def add_rotation_noise_to_first_robot(vel_cmd, theta):
    """
    只对每个 batch 中的第一个机器人（索引0）添加旋转干扰
    vel_cmd: [batch, robots, 2]
    theta: 干扰旋转角（弧度），可以是标量或 [batch] 每个样本不同角度
    """
    # 确保 theta 可以广播到 [batch, 1]
    theta = torch.tensor(theta)
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # 复制原始指令
    vel_noisy = vel_cmd.clone()
    
    # 只对第一个机器人进行旋转
    vx_first = vel_cmd[:, 0, 0]  # [batch]
    vy_first = vel_cmd[:, 0, 1]  # [batch]
    
    vx_rot = vx_first * cos_t.squeeze() - vy_first * sin_t.squeeze()
    vy_rot = vx_first * sin_t.squeeze() + vy_first * cos_t.squeeze()
    
    # 更新第一个机器人的速度
    vel_noisy[:, 0, 0] = vx_rot
    vel_noisy[:, 0, 1] = vy_rot
    
    return vel_noisy

# def add_rotation_to_obs(obs, theta):
#     """
#     对观测的位置和速度进行旋转
#     obs: [batch, agents, 4] 其中前4个元素是 [x, y, vx, vy]
#     theta: 旋转角（弧度），可以是标量或 [batch] 每个样本不同角度
#     """
#     theta = np.array(theta)
    
#     cos_t = np.cos(theta)
#     sin_t = np.sin(theta)
    
#     # 复制原始观测
#     obs_rotated = obs.copy()
    
#     # 分离位置和速度
#     x = obs_rotated[:, :, 0]  # [batch, agents]
#     y = obs_rotated[:, :, 1]  # [batch, agents]
#     vx = obs_rotated[:, :, 2]  # [batch, agents]
#     vy = obs_rotated[:, :, 3]  # [batch, agents]
    

#     x_rot = x * cos_t - y * sin_t
#     y_rot = x * sin_t + y * cos_t
#     vx_rot = vx * cos_t - vy * sin_t
#     vy_rot = vx * sin_t + vy * cos_t
#     #print(x[0][0], y[0][0],x_rot[0][0],y_rot[0][0])
#     #print(x_rot.shape)
#     # 更新观测
#     obs_rotated[:, :, 0] = x_rot
#     obs_rotated[:, :, 1] = y_rot
#     obs_rotated[:, :, 2] = vx_rot
#     obs_rotated[:, :, 3] = vy_rot
    
#     return obs_rotated

class OnPolicyMAAttackRunner(OnPolicyBaseRunner):

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
    def eval(self,episodes,attack_method='none', noise_level=0.1,noise_num=0.1):
        """Evaluate the model."""
        print("Evaluate the model.")
        device_id = 7  # 想用第2张卡就写1，第8张卡写7，以此类推
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device)
        
        path = self.model_path
        if self.flag:
            load_name = path + 'actor_agent' + str(0) + '.pt'
            self.actor[0].actor = torch.load(load_name,weights_only=False)
            self.actor[0].actor.n_threads = 1
            #self.actor[0].actor.update_edges()
        else:
            for agent_id in range(self.num_agents):
                load_name = path + 'actor_agent' + str(agent_id) + '.pt'
                self.actor[agent_id].actor = torch.load(load_name,weights_only=False)
        value_normalizer_state_dict = torch.load(path+'value_normalizer.pt',weights_only=False)
        self.value_normalizer.load_state_dict(value_normalizer_state_dict)
        #print(f'type of critic: {type(self.critic)}')
        critic_state_dict = torch.load(path+'critic_agent.pt',weights_only=False)
        # 使用 strict=False 来处理模型结构不匹配的问题
        missing_keys, unexpected_keys = self.critic.critic.load_state_dict(critic_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
        
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
        while True:
            #print(eval_episode)
            self.actor[0].actor.zero_grad()
            #print(eval_obs.shape)
            if attack_method == 'obs_grd_all' or attack_method == 'obs_grd_single' or attack_method == 'obs_grd_single_percp' or attack_method == 'obs_grd_all_percp':
                obs_tensor = torch.from_numpy(eval_obs).float().to(self.actor[0].device).requires_grad_(True)

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
            #print(len(eval_obs_list))
            #print(np.stack(eval_obs_list, axis=0).transpose(1, 0, 2).shape)  #10*10*34
            if attack_method == 'obs_grd_all' or attack_method == 'obs_grd_single' or attack_method == 'obs_grd_single_percp' or attack_method == 'obs_grd_all_percp':
                eval_actions, temp_rnn_state = self.actor[0].act(
                    obs_tensor,
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
            else:
                eval_actions, temp_rnn_state = self.actor[0].act(
                    np.stack(eval_obs_list, axis=0).transpose(1, 0, 2),
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
            #print(eval_actions.shape)
            if attack_method == 'obs_grd_all' or attack_method == 'obs_grd_single' or attack_method == 'obs_grd_single_percp' or attack_method == 'obs_grd_all_percp':
                if attack_method == 'obs_grd_all' or attack_method == 'obs_grd_all_percp':
                    target_actions = eval_actions
                elif attack_method == 'obs_grd_single' or attack_method == 'obs_grd_single_percp':
                    eval_actions = eval_actions.reshape(self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 2)
                    target_actions = eval_actions[:,0,:]  # only the first robot is attacked
                action_loss = -torch.norm(target_actions, p=2)  # maximize the L2 norm of the action
                #action_loss = target_actions.mean()  # maximize the sum of the action values
                action_loss.backward()
                obs_grad = obs_tensor.grad.data
                #print(obs_grad[0][0])  
                if attack_method == 'obs_grd_single' or attack_method == 'obs_grd_single_percp':
                    obs_grad[:,1:,:] = 0.0  # only the first robot is attacked

                #print(f'obs_tensor: {obs_tensor[3][3][0:4]}')
                perturbation = noise_level * obs_grad.sign()  # FGSM attack 
                if attack_method == 'obs_grd_single_percp' or attack_method == 'obs_grd_all_percp':     
                    obs_perturbed = obs_tensor.clone()
                    obs_perturbed[:,:,4:] = obs_tensor[:,:,4:] + perturbation[:,:,4:]
                else:
                    obs_perturbed = obs_tensor + perturbation
                #print(f'{obs_tensor[0,0]} \n {obs_perturbed[0,0]}')
                #print(f'obs_perturbed: {obs_perturbed[3][3][0:4]}')
                obs_perturbed = torch.clamp(obs_perturbed, -1.0, 1.0) 
                #print(f'{obs_tensor[0,0]} \n {obs_perturbed[0,0]}')
                eval_obs_list = []
                for agent_id in range(self.num_agents):
                    eval_obs_list.append(obs_perturbed[:, agent_id])
                eval_actions, temp_rnn_state = self.actor[0].act(
                    obs_perturbed,
                    np.stack(eval_rnn_states_list, axis=0),
                    np.stack(eval_masks_list, axis=0),
                    np.stack(eval_available_actions_list, axis=0).transpose(1, 0, 2) 
                    if len(eval_available_actions_list) > 0
                    else None, 
                    deterministic=True,
                )
            
            self.actor[0].actor.zero_grad()
            #eval_actions.backward()
            #print(f'eval_actions shape0: {eval_actions.shape}')  
            #print(self.algo_args["eval"]["n_eval_rollout_threads"])
            eval_actions = eval_actions.reshape(self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 2)
            #print(f'eval_actions shape1: {eval_actions.shape}')  
            eval_rnn_states = _t2n(temp_rnn_state).transpose(1, 0, 2, 3)
            #print(f'orin eval_actions: {eval_actions[3][3]}')
            if attack_method == 'act_rotation_all':
                eval_actions = add_rotation_noise_all(eval_actions, noise_num)
            if attack_method == 'act_rotation_single':
                eval_actions = add_rotation_noise_to_first_robot(eval_actions, noise_num) 
            #####test######
            # if attack_method == 'obs_rotation_all':
            #     eval_actions = add_rotation_noise_all(eval_actions, noise_num)
            ###############
            
            #print(f'after attack eval_actions: {eval_actions[3][3]}')
            eval_actions = _t2n(eval_actions)
            #print(eval_actions[0][0])
            if attack_method == 'act_noise_all':
                noise = np.random.normal(0, noise_level, size=eval_actions.shape)
                eval_actions += noise
            if attack_method == 'act_noise_single':
                noise = np.random.normal(0, noise_level, size=eval_actions[:,0].shape)
                eval_actions[:,0] += noise
                            
                
            #print(f'eval_actions shape: {eval_actions.shape}')
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            
            if attack_method == 'obs_noise_all_percp':
                #noise = np.random.uniform(low=-noise_level, high=noise_level, size=eval_obs[:,:,4:].shape)
                noise = np.random.normal(0, 3*noise_level, size=eval_obs[:,:,4:].shape)
                eval_obs[:,:,4:] += noise
                eval_obs = np.clip(eval_obs, -1.0, 1.0)  # clip the observation to a reasonable range

            if attack_method == 'obs_noise_all':
                noise = np.random.normal(0, 3*noise_level, size=eval_obs.shape)
                eval_obs += noise
                eval_obs = np.clip(eval_obs, -1.0, 1.0)  # clip the observation to a reasonable range
            
            if attack_method == 'obs_noise_single':
                noise = np.random.normal(0, 3*noise_level, size=eval_obs[:,0,0:].shape)
                eval_obs[:,0,0:] += noise
                eval_obs = np.clip(eval_obs, -1.0, 1.0)  # clip the observation to a reasonable range
                
            if attack_method == 'obs_noise_single_percp':
                noise = np.random.normal(0, 3*noise_level, size=eval_obs[:,0,4:].shape)
                eval_obs[:,0,4:] += noise
                eval_obs = np.clip(eval_obs, -1.0, 1.0)  # clip the observation to a reasonable range
            # if attack_method == 'obs_rotation_all':
            #     eval_obs[:,:,0:4] = add_rotation_to_obs(eval_obs[:,:,0:4], noise_num)
            #     noise = np.random.normal(0, noise_level, size=eval_obs[:,0,0:31].shape)
            #     eval_obs[:,0,0:31] += noise
            #     eval_obs = np.clip(eval_obs, -1.0, 1.0) 

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
                    #print(f"Evaluation episode {eval_episode} done.")
                    #print(np.sum(self.logger.one_episode_rewards[0][0], axis=0))
                    eval_rewards = np.sum(self.logger.one_episode_rewards[eval_i], axis=0)
                    #print('-')
                    print(f'episode {eval_episode} reward: {eval_rewards[0][0]}')
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
        return total_rewards / episodes