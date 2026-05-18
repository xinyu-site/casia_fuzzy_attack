import copy
import logging
from metadrive import (
    MultiAgentMetaDrive,
    MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv,
    MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv,
    MultiAgentParkingLotEnv,
)
import numpy as np

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class MetaDriveEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        # self.env = ss.pad_action_space_v0(ss.pad_observations_v0(self.module.parallel_env(**self.args)))
        envs_classes = dict(
            roundabout=MultiAgentRoundaboutEnv,
            intersection=MultiAgentIntersectionEnv,
            tollgate=MultiAgentTollgateEnv,
            bottleneck=MultiAgentBottleneckEnv,
            parkinglot=MultiAgentParkingLotEnv,
            pgma=MultiAgentMetaDrive,
        )
        self.env = envs_classes[args["scenario"]](dict(self.args["conf"]))
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = [f"agent{i}" for i in range(self.n_agents)]
        self.share_observation_space = self.unwrap(self.env.observation_space)
        self.observation_space = self.unwrap(self.env.observation_space)
        self.action_space = self.unwrap(self.env.action_space)

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        # print(len(obs))
        dones = {agent: agent not in term or term[agent] or trunc[agent] for agent in self.agents}
        return (
            self.unwrap(obs),
            self.unwrap(obs),
            self.corp_reward(self.unwrap(rew)),
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        obs = self.unwrap(self.env.reset()[0])
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
        l = []
        for agent in self.agents:
            # d是dict，如果agent不在d中，则按照d的第一个key的value来填充
            if agent not in list(d.keys()):
                # print("agent not in d.keys()", agent)
                agt = list(d.keys())[0]
                # 将值全部填充为0
                l.append(np.zeros_like(d[agt]))
            else:
                l.append(d[agent])
        return l
    
    def corp_reward(self, rew):
        return self.repeat(np.mean(rew))

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def seed(self, seed):
        self.env.seed(seed)
