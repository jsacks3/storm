from .model_base import DynamicsModelBase
import numpy as np
import torch
import gym

import mj_envs
import mjmpc.envs
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjmpc.utils import helpers

class MJModel(DynamicsModelBase):
    def __init__(self, model_params, tensor_args={'device':'cpu','dtype':torch.float32}):
        super(MJModel, self).__init__()
        self.model_params = model_params
        self.tensor_args = tensor_args

        self.num_cpu = self.model_params['mppi']['num_cpu']
        self.sim_env = SubprocVecEnv([self._make_env for i in range(self.num_cpu)])

        self.d_action = self.sim_env.action_space.shape[0]
        self.action_lows = self.sim_env.action_space.low
        self.action_highs = self.sim_env.action_space.high

    def _make_env(self):
        env_name = self.model_params['model']['env_name']
        gym_env = gym.make(env_name)
        rollout_env = GymEnvWrapper(gym_env)
        rollout_env.real_env_step(False)
        return rollout_env

    def rollout_open_loop(self, start_state, mean_act, delta):
        self.sim_env.set_env_state(start_state)

        num_particles, horizon, _ = delta.shape
        mean_act = mean_act.cpu().numpy()
        delta = delta.cpu().numpy()

        obs_vec, rew_vec, act_vec, done_vec, info_vec, next_obs_vec = self.sim_env.rollout(num_particles,
                                                                                           horizon,
                                                                                           mean_act.copy(),
                                                                                           delta,
                                                                                           mode='open_loop')

        trajectories = dict(
            actions = torch.tensor(act_vec, **self.tensor_args),
            costs = torch.tensor(-rew_vec, **self.tensor_args),
            rollout_time=0.0,
            state_seq = torch.tensor(obs_vec, **self.tensor_args)
        )

        return trajectories

    def get_next_state(self, curr_state, act, dt):
        pass