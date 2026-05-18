import os
from matplotlib.style import available
import numpy as np
from amb.envs.voltage.var_voltage_control.voltage_control_env import VoltageControl
from gym.spaces import Box

class ValtageEnv:
    def __init__(self, env_args) -> None:
        env_config_dict = env_args["env_args"]
        data_path = os.path.join(os.path.dirname(__file__), env_config_dict["data_path"])
        env_config_dict["data_path"] = data_path
        self.env = VoltageControl(env_config_dict)

        # an positive integer to reflect the number of agents.
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions

        act_space = Box(low=-env_config_dict["action_scale"]+env_config_dict["action_bias"], high=env_config_dict["action_scale"]+env_config_dict["action_bias"], shape=(self.n_actions,), dtype=np.float32)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        # spaces are a list of gym.Spaces, and the length of spaces equals to the number of agents.
        for _ in range(self.n_agents):
            self.action_space.append(act_space)
            self.observation_space.append([self.env.get_obs_size()])
            self.share_observation_space.append([self.env.get_state_size()])

    def seed(self, seed):
        """Set the random seed of this environment. Called before reset()."""
        np.random.seed(seed)
        self.env.reset()

    def reset(self):
        """Reset the environment.
        obs and share_obs are numpy.ndarrays that have shape (num_agents, vshape).
        available_actions are 0-1 numpy.ndarrays that have shape (num_agents, action_num) for discrete action spaces,
        and None for continuous action spaces.
        """
        obs, state = self.env.reset()
        return obs, [state] * self.n_agents, self.env.get_avail_actions()

    def step(self, actions):
        """Process a step of the environment.
        actions: numpy.ndarray (num_agents, action_shape). Actions must be 2-dimentional.

        obs, share_obs: numpy.ndarray (num_agents, vshape)
        rewards: numpy.ndarray (num_agents, 1). Rewards for different agents can be different.
        dones: boolean numpy.ndarray (num_agents, 1). True when an episode is done or the time is out of limit, else False.
        infos: list of dict, e.g., [{}, {}]
        available_actions: 0-1 numpy.ndarray (num_agents, action_num) or None.
        """
        # 去掉第二维度
        obs = np.array(self.env.get_obs())
        share_obs = np.array([self.env.get_state()] * self.n_agents)
        actions = actions.squeeze(1)
        # print(f"actions.shape: {actions.shape}", actions)
        reward, termiated, info = self.env.step(actions)
        rewards = np.array([[reward]] * self.n_agents)
        dones = np.array([termiated] * self.n_agents)
        infos = [info] * self.n_agents
        avail_actions = np.array(self.env.get_avail_actions())
        return obs, share_obs, rewards, dones, infos, avail_actions
    
    def render(self, mode):
        """render for environment.
        When mode is rgb_array, should return the rendered environment image in RGB values.
        """
        return self.env.render(mode)
        
    def close(self):
        """Close the environment and release the resources."""
        return
