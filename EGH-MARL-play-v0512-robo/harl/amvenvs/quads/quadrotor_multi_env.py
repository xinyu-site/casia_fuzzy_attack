import copy
import logging
import numpy as np
import argparse
from sample_factory.envs.create_env import create_env
from sample_factory.envs.env_utils import register_env

from amb.envs.quads.env_wrappers.quad_utils import make_quadrotor_env
from amb.envs.quads.env_wrappers.quadrotor_params import add_quadrotors_env_args

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def dict_to_object(input_dict):
    class DictToObjectNamespace(argparse.Namespace):
        def __init__(self, input_dict):
            super().__init__(**input_dict)

    return DictToObjectNamespace(input_dict)


class QuadrotorMultiEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        register_env("quadrotor_multi", make_quadrotor_env)
        self.env = create_env("quadrotor_multi", cfg=dict_to_object(self.args["conf"]))
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = [f"agent{i}" for i in range(self.n_agents)]
        self.share_observation_space = self.unwrap(self.env.observation_space)
        self.observation_space = self.unwrap(self.env.observation_space)
        self.action_space = self.unwrap(self.env.action_space)
        self._seed = 0

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, term, trunc, info = self.env.step(actions)
        dones = term + trunc
        return (
            obs,
            obs,
            rew,
            dones,
            info,
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        obs = self.env.reset(seed=self._seed)[0]
        return obs, obs, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = [d for _ in range(self.n_agents)]
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def seed(self, seed):
        self._seed = seed
