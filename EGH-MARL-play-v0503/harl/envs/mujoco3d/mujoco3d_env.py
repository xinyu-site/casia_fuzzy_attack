import gym
import os
from gym.spaces import Box
import numpy as np
from harl.envs.mujoco3d.utils import registerEnvs, getGraphStructure, getMotorJoints, getAdjacency, getGraphJoints
from harl.envs.mujoco3d.wrappers import ModularEnvWrapper
from .multiagentenv import MultiAgentEnv


class Mujoco3D(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.episode_limit = kwargs["env_args"]["episode_limit"]
        self.observation_graph_type = kwargs["env_args"]["observation_graph_type"]
        self.local_mode = kwargs["env_args"]["local_mode"]

        self.timestep = 0

        custom_xml = os.path.join(self.scenario, kwargs["env_args"]["custom_xml"]) 
        assert ".xml" in os.path.basename(custom_xml), "No XML file found."
        name = os.path.basename(custom_xml)
        env_name = name[:-4]
        self.graph = getGraphStructure(custom_xml, self.observation_graph_type)
        self.n_agents = len(self.graph)
        self.adj = getAdjacency(self.graph)

        self.limb_obs_size, self.limb_action_size, self.max_action, self.share_observation_space, self.action_space = registerEnvs(
            env_name, self.episode_limit, custom_xml
        )

        if self.local_mode:
            self.limb_obs_size += self.n_agents
            self.share_observation_space = Box(low=-np.inf, high=np.inf, shape=(self.limb_obs_size * self.n_agents, ))
        
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.limb_obs_size,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            self.share_observation_space for _ in range(self.n_agents)
        ]
        self.action_space = [
            Box(low=-1, high=1, shape=(self.limb_action_size,))
            for _ in range(self.n_agents)
        ]
        self.env = gym.make("harl.envs.mujoco3d.environments:%s-v0" % env_name)
        # self.env = ModularEnvWrapper(env)

        # match the order of modular policy actions to the order of environment actions
        self.motors = getMotorJoints(self.env.xml)
        self.joints = getGraphJoints(self.env.xml)
        self.action_order = [-1, -1, -1] * self.n_agents
        for i in range(1,len(self.joints)):
            self.action_order[3*i] = self.motors.index(self.joints[i][1])
            self.action_order[3*i+1] = self.motors.index(self.joints[i][2])
            self.action_order[3*i+2] = self.motors.index(self.joints[i][3])


    def step(self, actions):
        self.timestep += 1
        # clip the 0-padding before processing
        actions = actions.reshape(-1)
        action = actions[: self.n_agents * 3]
        # match the order of the environment actions
        env_action = [None for i in range(len(self.motors))]
        for i in range(len(action)):
            if self.action_order[i] != -1:
                env_action[self.action_order[i]] = action[i]
        state, reward, done, info = self.env.step(env_action)
        obs = state.reshape(self.n_agents, -1)
        if self.timestep >= self.episode_limit:
            done = True
        if self.local_mode:
            obs = np.concatenate([obs, self.adj], axis=-1)
            state = obs.reshape(-1)
        return (
            obs, 
            self.repeat(state), 
            np.full((self.n_agents, 1), reward), 
            self.repeat(done), 
            self.repeat(info),
            self.get_avail_actions()
        )

    def get_avail_actions(self):
        return None

    def reset(self):
        self.timestep = 0
        state = self.env.reset()
        obs = state.reshape(self.n_agents, -1)
        if self.local_mode:
            obs = np.concatenate([obs, self.adj], axis=-1)
            state = obs.reshape(-1)
        return obs, self.repeat(state), self.get_avail_actions()

    def render(self, mode='rgb_array'):
        img = self.env.render(mode)
        return img

    def close(self):
        self.env.close()

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
