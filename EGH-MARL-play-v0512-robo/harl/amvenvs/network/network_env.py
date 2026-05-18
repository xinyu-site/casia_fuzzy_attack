
import os

import numpy as np
from amb.envs.network.envs.large_grid_env import LargeGridEnv
import configparser
from gym.spaces import Discrete

from amb.envs.network.envs.real_net_env import RealNetEnv
from amb.envs.network.envs.CACC import CACCWrapper


class NetworkEnv:
    def __init__(self, env_args, port=0) -> None:
        ncfg = env_args["network_cfg"]
        ncfg_path = os.path.join(os.path.dirname(__file__), "config", ncfg)
        scenario = env_args["scenario"]
        self.scenario = scenario
        config = configparser.ConfigParser()
        config.read(ncfg_path)
        # update config["ENV_CONFIG"] with extra_args
        if "override_args" in env_args and env_args["override_args"]:
            for k, v in env_args["override_args"].items():
                if k in config["ENV_CONFIG"]:
                    print(f"update config {k}: {config['ENV_CONFIG'][k]} -> {v}")
                    config["ENV_CONFIG"][k] = str(v)

        if scenario == "grid":
            self.env = LargeGridEnv(config["ENV_CONFIG"], port=port, output_path=env_args["output_dir"], is_record=True, record_stat=True)
        elif scenario == "net":
            self.env = RealNetEnv(config["ENV_CONFIG"], port=port, output_path=env_args["output_dir"], is_record=True, record_stat=True)
        elif scenario == "catchup":
            self.env = CACCWrapper(config["ENV_CONFIG"], bias=0, std=100)
        elif scenario == "slowdown":
            self.env = CACCWrapper(config["ENV_CONFIG"], bias=0, std=100)
        else:
            raise NotImplementedError(f"Scenario {scenario} not implemented.")
    
        # self.env.reset()
        self.n_agents = self.env.n_agent
        self.n_s_ls = self.env.n_s_ls
        self.n_a_ls = self.env.n_a_ls

        self.observation_space = np.array(self.n_s_ls).reshape(len(self.n_s_ls), 1).tolist()
        self.share_observation_space = np.array(self.n_s_ls).reshape(len(self.n_s_ls), 1).tolist()
        self.action_space = [Discrete(value) for value in self.n_a_ls]

        # 针对net的环境需要单独判断，share_observation_space 要处理成最大长度对齐，并保持原有的维度
        if scenario == "net":
            self.share_observation_space = np.array([max(self.n_s_ls)] * len(self.n_s_ls)).reshape(len(self.n_s_ls), 1).tolist()


        # print(f"n_agents: {self.n_agents}")
        # print(f"action space: {self.action_space}")
        # print(f"observation space: {self.observation_space}")
        # print(f"share_observation_space: {self.share_observation_space}")
        # print(f"obs class", type(self.observation_space.__class__.__name__))
        # print(f"available_actions: ", self.get_avail_actions())


    def step(self, actions):
        """Process a step of the environment.
        actions: numpy.ndarray (num_agents, action_shape). Actions must be 2-dimentional.

        obs, share_obs: numpy.ndarray (num_agents, vshape)
        rewards: numpy.ndarray (num_agents, 1). Rewards for different agents can be different.
        dones: boolean numpy.ndarray (num_agents, 1). True when an episode is done or the time is out of limit, else False.
        infos: list of dict, e.g., [{}, {}]
        available_actions: 0-1 numpy.ndarray (num_agents, action_num) or None.
        """
        obs, reward, done, global_reward = self.env.step(actions)
        rewards = np.array([global_reward] * self.n_agents)
        dones = np.array([done] * self.n_agents)
        infos = [{}] * self.n_agents

        # 此处也需要对net的环境进行处理，将obs对齐到最大长度，后面补0，赋值给share_obs
        if self.scenario == "net":
            share_obs = []
            for i in range(self.n_agents):
                share_obs.append(np.array(np.pad(obs[i], ((0, max(self.n_s_ls) - len(obs[i]))), 'constant')))
        else:
            share_obs = obs

        # obs, share_obs, rewards, dones, infos, available_actions
        return obs, share_obs, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        obs = self.env.reset()

        if self.scenario == "net":
            share_obs = []
            for i in range(self.n_agents):
                share_obs.append(np.array(np.pad(obs[i], ((0, max(self.n_s_ls) - len(obs[i]))), 'constant')))
        else:
            share_obs = obs

        # print("===>obs: ", obs)
        # print("===>share_obs: ", share_obs)

        return obs, share_obs, self.get_avail_actions()

    def seed(self, seed):
        self.env.seed = seed
        self.env.reset()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n
    
    def close(self):
        
        return
