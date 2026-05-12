import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check, get_shape_from_obs_space, get_shape_from_act_space
from torch.distributions.normal import Normal
from torch.distributions import Bernoulli
from harl.models.base.model_util import *
from harl.envs.ma_envs.envs.point_envs.local_obs_util import *
from harl.models.base.eghn_v2 import EGNNv2
# from harl.models.base.eghn_v2 import EGAN
from harl.models.base.distributions import FixedCategorical
from harl.utils.models_tools import init, get_init_method

class EgnnV3Policy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(EgnnV3Policy, self).__init__()
        self.env_name = args["env_name"]
        self.hidden_nf = args["hidden_sizes"]
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        # EQC baseline
        self.use_eqc_flag = args["use_eqc_flag"]
        self.subgroup_num = int(args["subgroup_num"])
        self.rnn_mean_flag = args["rnn_mean_flag"]
        
        self.n_nodes = args["num_agents"]
        self.n_threads = args["n_rollout_threads"]
        self.episode_length = args["episode_length"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.mini_batch_size = self.n_threads * self.episode_length // self.actor_num_mini_batch
        
        # env args
        self.env_name = args["env_name"]
        obs_shape = get_shape_from_obs_space(obs_space)
        act_dim = get_shape_from_act_space(action_space)
        self.flat = args["flat"]
        self.in_edge_nf = args["in_edge_nf"]
        self.n_layers = args["n_layers"]
        self.dimension = args["dimension"]
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        # 时序信息处理
        self.use_history = args["use_history"]
        self.windows_size = args["windows_size"]
        self.c = 1

        # 局部信息处理
        self.local_info_input = args["local_info_input"]
        self.local_info_output = args["local_info_output"]
        self.hpn_hidden = args["hpn_hidden"]
        self.local_mode = args["local_mode"]
        self.use_hpn = args["use_hpn"]
        self.local_tool = local_tool(self, args=args, net_type="egnn", id="actor")
        self.local_tool.dim_info_init(obs_shape[0])
        self.n_vector_input = self.local_tool.equ_nf // self.dimension

        if self.local_mode:
            self.local_module = nn.ModuleList()
            for i in range(len(self.local_info_output)):
                if self.use_hpn:
                    layer = HyperMLP(input_dim=self.local_info_input[i], output_dim=self.local_info_output[i], hyper_hidden_dim=self.hpn_hidden[i])
                else:
                    layer = LocalMLP(input_dim=self.local_info_input[i], output_dim=self.local_info_output[i])
                self.local_module.append(layer)
        else:
            self.local_module = None
            
        self.egnn_model = EGNNv2(
            n_layers=self.n_layers,
            in_edge_nf=self.in_edge_nf,
            in_node_nf=self.local_tool.inv_nf_new,
            hidden_nf=self.hidden_nf[0],
            n_vector_input=self.n_vector_input,
            device=device,
            flat=self.flat,
            activation=nn.SiLU(),
            norm=True
        )

        # self.egnn_model = EGAN(
        #     n_layers=self.n_layers, 
        #     in_edge_nf=self.in_edge_nf,
        #     in_node_nf=self.local_tool.inv_nf_new, 
        #     hidden_nf=self.hidden_nf[0], 
        #     n_vector_input=self.n_vector_input,
        #     num_heads=2,
        #     embed_size=128,
        #     device=device,
        #     flat=self.flat, 
        #     activation=nn.SiLU(), 
        #     norm=True
        # )

        self.act_net = nn.Linear(self.n_vector_input, 1, bias=False)
        if self.env_name == "mujoco3d":
            self.act_net2 = nn.Linear(self.hidden_nf[-1] + self.dimension, self.dimension)
        elif self.env_name == "smacv2":
            self.move_templates = nn.Parameter(torch.tensor([
                [0, 1],     # up
                [0, -1],    # down
                [1, 0],      # right
                [-1, 0],    # left
            ], dtype=torch.float), requires_grad=False)  # 固定方向模板
            # self.direction_logit_scale = nn.Parameter(torch.tensor(1.0))
            
            init_method = get_init_method(self.initialization_method)
            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), self.gain)

            act_dim = action_space.n
            self.linear = init_(nn.Linear(self.hidden_nf[-1], act_dim - 4))  # 生成非方向动作
            self.move_decider = init_(nn.Linear(self.hidden_nf[-1], 1))  # 决定是移动还是非移动
            # 非移动动作的映射表
            self.register_buffer('non_move_mapping', torch.tensor([0, 1, 6, 7, 8, 9, 10]))
            self.register_buffer('action_to_nonmove_idx', torch.tensor([0, 1, 0, 0, 0, 0, 2, 3, 4, 5, 6]))


        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.to(device)

    def update_local_tool(self, obs_space):
        self.local_tool = local_tool(self, args=self.args, net_type="egnn", id="actor")
        obs_shape = get_shape_from_obs_space(obs_space)
        self.local_tool.dim_info_init(obs_shape[0])

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        
        obs = self.local_tool.trans_info2local_actor(obs)
        obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.reshape(self.n_threads * self.n_nodes, -1)
        
        obs = self.local_tool.local_info_process(obs, self.local_module)
        
        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :self.dimension]

        equ_fea = equ_fea.reshape(equ_fea.shape[0], -1, self.dimension).transpose(1, 2)
        rows, cols = self.local_tool.forward_edges

        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()

        equ_out, h_out = self.egnn_model(equ_fea, h, self.local_tool.forward_edges, edge_fea=None)
        
        act = self.act_net(equ_out)  # 输出的等变特征进行线性组合，得到需要的维度

        if self.env_name == "mujoco3d":  # 保证动作在mujoco3d中保持不变性
            equ0 = equ_fea[:, :, 4:7]
            act0 = torch.bmm(equ0[:, :, 0:1].transpose(-2, -1), act).reshape(act.shape[0], -1)
            act1 = torch.bmm(equ0[:, :, 1:2].transpose(-2, -1), act).reshape(act.shape[0], -1)
            act2 = torch.bmm(equ0[:, :, 2:].transpose(-2, -1), act).reshape(act.shape[0], -1)
            # act = self.act_net2(torch.cat([act0, act1, act2, h_out], dim=-1))
            act = torch.cat([act0, act1, act2], dim=-1)
        elif self.env_name == 'smacv2':
            act = act.squeeze(-1)
            # === 生成各个部分的logits ===
            move_logits = torch.matmul(act, self.move_templates.T)
            # direction_logits = self.direction_logit_scale * direction_logits
            inv_logits = self.linear(h_out)
            move_decider_logits = self.move_decider(h_out).squeeze(-1)
            
            # ===获取两类动作的可行动作===
            available_move = available_actions[:, 2: 6]
            available_non_move = torch.cat([
                available_actions[:, :2],
                available_actions[:, 6:]
            ], dim=-1)
            
            # 屏蔽不可执行的动作
            move_logits = move_logits.masked_fill(available_move == 0, -1e10)
            inv_logits = inv_logits.masked_fill(available_non_move == 0, -1e10)

            # ===生成是否移动的概率分布===
            move_prob = torch.sigmoid(move_decider_logits)
            can_move = available_move.sum(dim=-1) > 0
            can_non_move = available_non_move.sum(dim=-1) > 0
            # 处理一类动作都不可执行的极端情况
            move_prob = torch.where(
                can_move & ~can_non_move,
                torch.ones_like(move_prob),
                torch.where(
                    ~can_move & can_non_move,
                    torch.zeros_like(move_prob),
                    move_prob
                )
            )
            move_decider_dist = Bernoulli(probs=move_prob)
            # 非确定性动作采样，确定性动作大于0.5移动
            if deterministic:
                is_move = (move_prob > 0.5).float()
            else:
                is_move = move_decider_dist.sample()
            
            # ===两类动作的概率分布===
            move_dist = FixedCategorical(logits=move_logits)
            non_move_dist = FixedCategorical(logits=inv_logits)

            if deterministic:
                move_action = move_dist.mode().squeeze(-1)
                non_move_action = non_move_dist.mode().squeeze(-1)
            else:
                move_action = move_dist.sample().squeeze(-1)
                non_move_action = non_move_dist.sample().squeeze(-1)
            
            # ===产生最终动作===
            true_move_action = move_action + 2
            true_non_move_action = self.non_move_mapping[non_move_action]
            
            actions = is_move * true_move_action + (1 - is_move) * true_non_move_action
            actions = actions.reshape(self.n_threads, self.n_nodes, -1)

            move_log_prob = move_dist.log_prob(move_action)
            non_move_log_prob = non_move_dist.log_prob(non_move_action)
            move_decider_log_prob = move_decider_dist.log_prob(is_move)
            action_log_probs = move_decider_log_prob + is_move * move_log_prob + (1 - is_move) * non_move_log_prob
            action_log_probs = action_log_probs.reshape(self.n_threads, self.n_nodes, -1)
            # x = torch.cat([
            #     inv_logits[:, :2],
            #     direction_logits,
            #     inv_logits[:, 2:]
            # ], dim=-1)
            # x = x.reshape(self.n_threads, self.n_nodes, -1)
            # x[available_actions == 0] = -1e10
            # action_distribution = FixedCategorical(logits=x)
            # actions = (
            #     action_distribution.mode()
            #     if deterministic
            #     else action_distribution.sample()
            # )
            # # 用于防止smacv2训练过程中采集到不可用动作
            # while True:
            #     valid = torch.gather(available_actions, dim=-1, index=actions) == 1
            #     if valid.all():
            #         break
            #     actions = action_distribution.sample()
            # action_log_probs = action_distribution.log_probs(actions)
            return actions, action_log_probs, rnn_states
        else:
            act = act.reshape(equ_out.shape[0], -1)

        mu = act
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        actions = policy.sample()
        action_log_probs = policy.log_prob(actions)
        actions = actions.reshape(self.n_threads, self.n_nodes, -1)
        action_log_probs = action_log_probs.reshape(self.n_threads, self.n_nodes, -1)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = self.local_tool.trans_info2local_actor(obs, evaluate_mode=True)

        obs = obs.reshape(-1, self.local_tool.inv_nf_old + self.local_tool.equ_nf)

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        obs = self.local_tool.local_info_process(obs, self.local_module)
        
        equ_fea = obs[:, :self.local_tool.equ_nf]
        h = obs[:, self.local_tool.equ_nf:]
        loc = equ_fea[:, :self.dimension]
        # vel = equ_fea[:, 2:]
        equ_fea = equ_fea.reshape(equ_fea.shape[0], -1, self.dimension).transpose(1, 2)
        
        rows, cols = self.local_tool.eval_edges
        if self.in_edge_nf == 0:
            edge_attr = None
        else:
            edge_attr = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1).detach()
        
        equ_out, h_out = self.egnn_model(equ_fea, h, self.local_tool.eval_edges, edge_fea=None)
        
        act = self.act_net(equ_out)
        if self.env_name == "mujoco3d":  # 保证动作在mujoco3d中保持不变性
            equ0 = equ_fea[:, :, 4:7]
            act0 = torch.bmm(equ0[:, :, 0:1].transpose(-2, -1), act).reshape(act.shape[0], -1)
            act1 = torch.bmm(equ0[:, :, 1:2].transpose(-2, -1), act).reshape(act.shape[0], -1)
            act2 = torch.bmm(equ0[:, :, 2:].transpose(-2, -1), act).reshape(act.shape[0], -1)
            # act = self.act_net2(torch.cat([act0, act1, act2, h_out], dim=-1))
            act = torch.cat([act0, act1, act2], dim=-1)
        elif self.env_name == 'smacv2':
            # act = act.squeeze(-1)
            # direction_logits = torch.matmul(act, self.direction_templates.T)
            # direction_logits = self.direction_logit_scale * direction_logits
            # inv_logits = self.linear(h_out)
            # x = torch.cat([
            #     inv_logits[:, :2],
            #     direction_logits,
            #     inv_logits[:, 2:]
            # ], dim=-1)
            # x[available_actions == 0] = -1e10
            # action_distribution = FixedCategorical(logits=x)
            # action_log_probs = action_distribution.log_probs(action)
            # if active_masks is not None:
            #     dist_entropy = (
            #         action_distribution.entropy() * active_masks.squeeze(-1)
            #     ).sum() / active_masks.sum()
            # else:
            #     dist_entropy = action_distribution.entropy().mean()
            act = act.squeeze(-1)
            # === 生成各个部分的logits ===
            move_logits = torch.matmul(act, self.move_templates.T)
            # direction_logits = self.direction_logit_scale * direction_logits
            inv_logits = self.linear(h_out)
            move_decider_logits = self.move_decider(h_out).squeeze(-1)
            
            # ===获取两类动作的可行动作===
            available_move = available_actions[:, 2: 6]
            available_non_move = torch.cat([
                available_actions[:, :2],
                available_actions[:, 6:]
            ], dim=-1)
            
            # 屏蔽不可执行的动作
            move_logits = move_logits.masked_fill(available_move == 0, -1e10)
            inv_logits = inv_logits.masked_fill(available_non_move == 0, -1e10)

            # ===生成是否移动的概率分布===
            move_prob = torch.sigmoid(move_decider_logits)
            can_move = available_move.sum(dim=-1) > 0
            can_non_move = available_non_move.sum(dim=-1) > 0
            # 处理一类动作都不可执行的极端情况
            move_prob = torch.where(
                can_move & ~can_non_move,
                torch.ones_like(move_prob),
                torch.where(
                    ~can_move & can_non_move,
                    torch.zeros_like(move_prob),
                    move_prob
                )
            )
            move_decider_dist = Bernoulli(probs=move_prob)
            
            # ===两类动作的概率分布===
            move_dist = FixedCategorical(logits=move_logits)
            non_move_dist = FixedCategorical(logits=inv_logits)

            is_move = ((action >= 2) & (action <= 5)).float().squeeze(-1)
            move_action = (action - 2).clamp(min=0, max=3).squeeze(-1)
            non_move_action = self.action_to_nonmove_idx[action.long()].squeeze(-1)

            # === log_probs ===
            move_log_prob = move_dist.log_prob(move_action)  # [B]
            non_move_log_prob = non_move_dist.log_prob(non_move_action)  # [B]
            move_decider_log_prob = move_decider_dist.log_prob(is_move)  # [B]

            final_log_prob = move_decider_log_prob + is_move * move_log_prob + (1 - is_move) * non_move_log_prob
            action_log_probs = final_log_prob.unsqueeze(-1)  # [B,1]

            # === Entropy ===
            move_entropy = move_dist.entropy()  # [B]
            non_move_entropy = non_move_dist.entropy()  # [B]
            move_decider_entropy = move_decider_dist.entropy()  # [B]

            action_entropy = move_prob * move_entropy + (1 - move_prob) * non_move_entropy
            dist_entropy = move_decider_entropy + action_entropy  # [B]
            if active_masks is not None:
                dist_entropy = (
                    dist_entropy * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
            else:
                dist_entropy = dist_entropy.mean()

            return action_log_probs, dist_entropy, None
        else:
            act = act.reshape(equ_out.shape[0], -1)
        mu = act
        std = torch.exp(self.log_std)
        policy = Normal(mu, std)
        action_distribution = policy
        dist_entropy = action_distribution.entropy().sum(axis=-1)
        dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        action_log_probs = policy.log_prob(action)
        return action_log_probs, dist_entropy, action_distribution
    
