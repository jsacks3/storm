import torch
import yaml
import numpy as np

from .task_base import BaseTask
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from ...util_file import get_mpc_configs_path as mpc_configs_path
from ..rollout.rollout_cartpole import CartpoleRollout
from ...mpc.control import MPPI

class CartpoleTask(BaseTask):
    def __init__(self, dynamics_model, task_file, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        super(CartpoleTask, self).__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(dynamics_model, task_file)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.num_dof

    def get_rollout_fn(self, **kwargs):
        rollout_fn = CartpoleRollout(**kwargs)
        return rollout_fn

    def init_mppi(self, dynamics_model, task_file):
        mpc_yml_file = join_path(mpc_configs_path(), task_file)
        with open(mpc_yml_file) as file:
            exp_params = yaml.load(file, Loader=yaml.FullLoader)

        rollout_fn = self.get_rollout_fn(dynamics_model=dynamics_model, tensor_args=self.tensor_args)

        mppi_params = exp_params['mppi']
        model_params = exp_params['model']
        dynamics_model = rollout_fn.dynamics_model

        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -model_params['max_torque'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        mppi_params['action_highs'] = model_params['max_torque'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        init_action = torch.zeros((mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        controller = MPPI(**mppi_params)
        self.exp_params = exp_params
        return controller

    def _state_to_tensor(self, state):
        state_tensor = np.concatenate((state['position'], state['velocity']))
        state_tensor = torch.tensor(state_tensor, **self.tensor_args)
        return state_tensor

    def get_command(self, t_step, curr_state, control_dt, WAIT=False):
        state_tensor = self._state_to_tensor(curr_state).unsqueeze(0)
        command, val, info = self.controller.optimize(state_tensor)
        return command[0]