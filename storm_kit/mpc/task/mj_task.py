import torch
import yaml
import numpy as np

from .task_base import BaseTask
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from ...util_file import get_mpc_configs_path
from ..rollout.mj_rollout import MJRollout
from ...mpc.control import MPPI
from ..model.mj_model import MJModel

class MJTask(BaseTask):
    def __init__(self, task_file, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        super(MJTask, self).__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(task_file)

    def get_rollout_fn(self, **kwargs):
        rollout_fn = MJRollout(**kwargs)
        return rollout_fn

    def init_mppi(self, task_file):
        task_params = load_yaml(join_path(get_mpc_configs_path(), task_file))
        dynamics_model = MJModel(model_params=task_params, tensor_args=self.tensor_args)
        rollout_fn = self.get_rollout_fn(dynamics_model=dynamics_model, tensor_args=self.tensor_args)

        mppi_params = task_params['mppi']
        model_params = task_params['model']
        dynamics_model = rollout_fn.dynamics_model

        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = torch.tensor(dynamics_model.action_lows, **self.tensor_args)
        mppi_params['action_highs'] = torch.tensor(dynamics_model.action_highs, **self.tensor_args)

        init_action = torch.zeros((mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args

        if 'num_cpu' and 'particles_per_cpu' in mppi_params:
            mppi_params['num_particles'] = mppi_params['num_cpu'] * mppi_params['particles_per_cpu']

        mppi_params.pop('particles_per_cpu', None)
        mppi_params.pop('num_cpu', None)

        controller = MPPI(**mppi_params)
        self.task_params = task_params
        return controller

    def get_command(self, curr_state):
        command, val, info = self.controller.optimize(curr_state)
        return command[0]