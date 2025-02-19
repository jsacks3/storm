import torch
import yaml
import numpy as np

from .task_base import BaseTask
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from ...util_file import get_mpc_configs_path
from ..rollout.cartpole_isaac_rollout import CartpoleIsaacRollout
from ...mpc.control import MPPI
from ..model.cartpole_isaac_model import CartpoleIsaacModel

class CartpoleIsaacTask(BaseTask):
    def __init__(self, env, task_file, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        super(CartpoleIsaacTask, self).__init__(tensor_args=tensor_args)
        self.env = env
        self.gym_instance = env.gym_instance
        self.controller = self.init_mppi(task_file)

    def get_rollout_fn(self, **kwargs):
        rollout_fn = CartpoleIsaacRollout(**kwargs)
        return rollout_fn

    def init_mppi(self, task_file):
        task_params = load_yaml(join_path(get_mpc_configs_path(), task_file))
        dynamics_model = CartpoleIsaacModel(gym_instance=self.gym_instance, env=self.env, tensor_args=self.tensor_args)
        rollout_fn = self.get_rollout_fn(dynamics_model=dynamics_model, tensor_args=self.tensor_args)

        mppi_params = task_params['mppi']
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = dynamics_model.action_lows
        mppi_params['action_highs'] = dynamics_model.action_highs

        init_action = torch.zeros((mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args

        controller = MPPI(**mppi_params)
        self.task_params = task_params
        return controller

    def get_command(self, curr_state):
        command, val, info = self.controller.optimize(curr_state)
        return command[0]