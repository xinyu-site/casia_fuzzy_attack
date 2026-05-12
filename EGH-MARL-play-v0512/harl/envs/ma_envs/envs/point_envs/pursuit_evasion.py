import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from harl.envs.ma_envs.commons.utils import EzPickle
from harl.envs.ma_envs import base
# from ma_envs.envs.environment import MultiAgentEnv
from harl.envs.ma_envs.agents.point_agents.pursuer_agent import PointAgent
from harl.envs.ma_envs.agents.point_agents.evader_agent import Evader
from harl.envs.ma_envs.commons import utils as U
import networkx as nwx
import itertools
try:
    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpla
    from matplotlib.patches import Wedge
    from matplotlib.patches import RegularPolygon
    import matplotlib.patches as patches
except:
    pass


class PursuitEvasionEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self,
                 local_mode=True,
                 obs_radius=100,
                 windows_size=1,
                 use_history=False,
                 nr_pursuers=5,
                 nr_evaders=1,
                #  distance_threshold=100,
                #  nearest_num=5,
                 obs_mode='2D_rbf',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=8,
                 bearing_bins=8,
                 torus=True,
                 dynamics='direct',
                 evader_policy='circle',
                 env_num1 = 0.0,
                 env_num2 = 0.0):
        EzPickle.__init__(self, nr_pursuers, nr_evaders, obs_mode, comm_radius, world_size, distance_bins,
                          bearing_bins, torus, dynamics)
        self.local_mode = local_mode
        self.obs_radius = obs_radius

        self.evader_policy = evader_policy
        self.nr_agents = nr_pursuers
        self.n_agents = self.nr_agents
        self.nr_evaders = 1
        self.obs_mode = obs_mode
        self.distance_bins = distance_bins
        self.bearing_bins = bearing_bins
        self.comm_radius = comm_radius
        self.obs_radius2 = comm_radius / 2
        self.torus = torus
        self.dynamics = dynamics
        self.world_size = world_size
        self.world = base.World(world_size, torus, dynamics, env_num1, env_num2)
        self.world.agents = [
            PointAgent(self) for _ in
            range(self.nr_agents)
        ]
        [self.world.agents.append(Evader(self)) for _ in range(self.nr_evaders)]
        self._reward_mech = 'global'
        # 使用时序数据
        self.use_history = use_history
        self.obs_his = U.obs_history(his_lenth=windows_size)
        
        self.timestep = None
        self.hist = None
        self.ax = None
        self.obs_comm_matrix = None
        self.obs_dim = self.agents[0].observation_space.shape[0]
        if self.obs_mode == 'sum_obs_learn_comm':
            self.world.dim_c = 1
        # self.seed()

    @property
    def share_observation_space(self):
        # dim = self.agents[0]._dim_o * self.nr_agents
        # return [spaces.Box(low=-np.inf, high=+np.inf, shape=(dim,), dtype=np.float32)] * self.nr_agents
        share_obs_space = {}
        shape = self.agents[0].observation_space.shape
        for agent_id in range(self.nr_agents):
            share_obs_space[agent_id] = spaces.Box(low=-np.float32(np.inf), high=np.float32(np.inf), 
                                                   shape=(shape[0]*self.nr_agents, ), dtype=np.float32)
        return share_obs_space

    @property
    def observation_space(self):
        # return [self.agents[0].observation_space] * self.nr_agents
        obs_space = {}
        for agent_id in range(self.nr_agents):
            obs_space[agent_id] = self.agents[agent_id].observation_space
        return obs_space

    def get_state(self,obs):
        share_obs = np.array(obs).reshape(1, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.nr_agents, axis=1)
        share_obs=share_obs.reshape(self.nr_agents,-1)
        return share_obs.copy()

    @property
    def state_space(self):
        return spaces.Box(low=-10., high=10., shape=(self.nr_agents * 3,), dtype=np.float32)

    @property
    def action_space(self):
        # return [self.agents[0].action_space] * self.nr_agents
        act_space = {}
        for agent_id in range(self.nr_agents):
            act_space[agent_id] = self.agents[agent_id].action_space
        return act_space

    @property
    def reward_mech(self):
        return self.reward_mech

    @property
    def agents(self):
        return self.world.policy_agents

    def get_param_values(self):
        return self.__dict__

    def seed(self, seed=None):
        # self.np_random, seed_ = seeding.np_random(seed)
        # return [seed_]
        np.random.seed(seed)

    @property
    def timestep_limit(self):
        return 512

    @property
    def is_terminal(self):
        if self.timestep >= self.timestep_limit:
            if self.ax:
                plt.close()
            return True
        return False
    
    def ue_set(self, status):
        states = status[0][0: self.n_agents]
        pursur_states = status[0][self.n_agents:]
        
        if self.use_history:
            self.obs_his.clear_obs()
        self.timestep = 0
        assert self.nr_agents == len(states)
        self.world.agents = [
            PointAgent(self)
            for _ in
            range(self.nr_agents)
        ]
        self.world.agents.append(Evader(self))
        self.obs_comm_matrix = self.obs_radius2 * np.ones([self.nr_agents + 1, self.nr_agents + 1])
        self.obs_comm_matrix[0:-self.nr_evaders, 0:-self.nr_evaders] = self.comm_radius

        # states:0-1:位置，2方位, 3-4速度;前面n-self.nr_evaders个表示追逐者，最后self.nr_evaders表示被追逐者
        pursuers = np.zeros((self.nr_agents, 3))
        pursuers[:, 0:2] = states[:, 0:2]
        pursuers[:, 2:3] = states[:, 2:3]
        evader = pursur_states[:, 0:2]

        self.world.agent_states = pursuers
        self.world.landmark_states = evader
        self.world.reset()
        
        for i, agent in enumerate(self.world.agents):
            agent.state.p_vel = status[0][i, 3:]
        velocities = np.vstack([agent.state.p_vel for agent in self.agents])
        
        feats = [p.graph_feature for p in self.agents]
        obs = []
        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     velocities
                                     )
            obs.append(ob)
        s_obs = np.array(obs)
        s_obs = np.reshape(s_obs, (self.obs_dim * self.nr_agents, ))
        s_obs = [s_obs for _ in range(self.nr_agents)]
        if self.use_history:
            self.obs_his.insert_obs((obs, s_obs))
            (obs, s_obs) = self.obs_his.get_obs()
        return obs, s_obs, None
    
    def reset(self):
        self.timestep = 0
        if self.use_history:
            self.obs_his.clear_obs()
        self.world.agents = [
            PointAgent(self)
            for _ in
            range(self.nr_agents)
        ]
        self.world.agents.append(Evader(self))
        self.obs_comm_matrix = self.obs_radius2 * np.ones([self.nr_agents + 1, self.nr_agents + 1])
        self.obs_comm_matrix[0:-self.nr_evaders, 0:-self.nr_evaders] = self.comm_radius
        pursuers = np.random.rand(self.nr_agents, 3)
        pursuers[:, 0:2] = self.world_size * ((0.95 - 0.05) * pursuers[:, 0:2] + 0.05)
        pursuers[:, 2:3] = 2 * np.pi * pursuers[:, 2:3]

        evader = (0.95 - 0.05) * np.random.rand(self.nr_evaders, 2) + 0.05
        evader = self.world_size * evader

        self.world.agent_states = pursuers
        self.world.landmark_states = evader
        self.world.reset()

        feats = [p.graph_feature for p in self.agents]
        obs = []

        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     np.zeros([self.nr_agents, 2])
                                     )
            obs.append(ob)
        s_obs = np.array(obs)
        s_obs = np.reshape(s_obs, (self.obs_dim * self.nr_agents, ))
        s_obs = [s_obs for _ in range(self.nr_agents)]
        if self.use_history:
            self.obs_his.insert_obs((obs, s_obs))
            (obs, s_obs) = self.obs_his.get_obs()
        return obs, s_obs, None

    def step(self, actions):
        self.timestep += 1
        assert len(actions) == self.nr_agents
        clipped_actions = np.clip(actions, self.agents[0].action_space.low, self.agents[0].action_space.high)
        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action[0:2]
            if self.world.dim_c > 0:
                agent.action.c = action[2:]
        self.world.step()
        feats = [p.graph_feature for p in self.agents]
        velocities = np.vstack([agent.state.w_vel for agent in self.agents])
        next_obs = []
        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     feats,
                                     velocities
                                     )
            next_obs.append(ob)
        rewards = self.get_reward(actions)
        done = self.is_terminal
        '''
        if rewards[0] > -1 / self.obs_radius2:  # distance of 1 in world coordinates, scaled by the reward scaling factor
            done = True
        '''
        info = {'pursuer_states': self.world.agent_states,
                'evader_states': self.world.landmark_states,
                'state': np.vstack([self.world.agent_states[:, 0:2], self.world.landmark_states]),
                'actions': actions}
        s_obs = np.array(next_obs)
        s_obs = np.reshape(s_obs, (self.obs_dim * self.nr_agents, ))
        s_obs = [s_obs for _ in range(self.nr_agents)]
        if self.use_history:
            self.obs_his.insert_obs((next_obs, s_obs))
            (next_obs, s_obs) = self.obs_his.get_obs()
        return next_obs, s_obs, rewards, done, info, None

    def get_reward(self, actions):
        r = -np.minimum(np.min(self.world.distance_matrix[-1, :-self.nr_evaders]), self.obs_radius2) / self.obs_radius2
        r = np.ones((self.nr_agents,1)) * r

        return r

    def graph_feature(self):
        adj_matrix = np.array(self.world.distance_matrix < self.obs_comm_matrix, dtype=float)
        sets = U.dfs(adj_matrix, 2)
        g = nwx.Graph()
        for set_ in sets:
            l_ = list(set_)
            if self.nr_agents in set_:
                dist_matrix = np.array([self.world.distance_matrix[x] for x in list(itertools.product(l_, l_))]).reshape(
                        [len(l_), len(l_)])
                obs_comm_matrix = np.array(
                    [self.obs_comm_matrix[x] for x in list(itertools.product(l_, l_))]).reshape(
                    [len(l_), len(l_)])
                adj_matrix_sub = np.array((0 <= dist_matrix) & (dist_matrix < obs_comm_matrix), dtype=float)
                connection = np.where(adj_matrix_sub == 1)
                edges = [[x[0], x[1]] for x in zip([l_[c] for c in connection[0]], [l_[c] for c in connection[1]])]

                g.add_nodes_from(l_)
                g.add_edges_from(edges)
                for ind, e in enumerate(edges):
                    g[e[0]][e[1]]['weight'] = dist_matrix[connection[0][ind], connection[1][ind]]

        for i in range(self.nr_agents):
            try:
                self.agents[i].graph_feature = \
                    nwx.shortest_path_length(g, source=i, target=self.nr_agents, weight='weight')
            except:
                self.agents[i].graph_feature = np.inf

        return sets

    def render(self, mode='human'):
        if mode == 'animate':
            output_dir = "/tmp/video/"
            if self.timestep == 0:
                import shutil
                import os
                try:
                    shutil.rmtree(output_dir)
                except FileNotFoundError:
                    pass
                os.makedirs(output_dir, exist_ok=True)

        if not self.ax:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.set_xlim((0, self.world_size))
            ax.set_ylim((0, self.world_size))
            self.ax = ax

        else:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.set_xlim((0, self.world_size))
            self.ax.set_ylim((0, self.world_size))

        comm_circles = []
        obs_circles = []
        self.ax.scatter(self.world.landmark_states[:, 0], self.world.landmark_states[:, 1], c='r', s=20)
        self.ax.scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=20)
        for i in range(self.nr_agents):
            comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                       self.world.agent_states[i, 1]),
                                      self.comm_radius, color='g', fill=False))
            self.ax.add_artist(comm_circles[i])

            obs_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.obs_radius2, color='g', fill=False))
            self.ax.add_artist(obs_circles[i])

        if mode == 'human':
            plt.pause(0.01)
        elif mode == 'animate':
            if self.timestep % 1 == 0:
                plt.savefig(output_dir + format(self.timestep//1, '04d'))

            if self.is_terminal:
                import os
                os.system("ffmpeg -r 10 -i " + output_dir + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4")


if __name__ == '__main__':
    nr_pur = 10
    env = PursuitEvasionEnv(nr_pursuers=nr_pur,
                            nr_evaders=1,
                            obs_mode='sum_obs_no_ori',
                            comm_radius=200 * np.sqrt(2),
                            world_size=100,
                            distance_bins=8,
                            bearing_bins=8,
                            dynamics='unicycle',
                            torus=True)
    for ep in range(1):
        o = env.reset()
        dd = False
        for t in range(1024):
            a = 1 * np.random.randn(nr_pur, env.world.agents[0].dim_a)
            a[:, 0] = 1
            o, rew, dd, _ = env.step(a)
            if t % 1 == 0:
                env.render()

            if dd:
                break
