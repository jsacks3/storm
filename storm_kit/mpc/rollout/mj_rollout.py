from .rollout_base import RolloutBase
import torch
import numpy as np

class MJRollout(RolloutBase):
    def __init__(self, dynamics_model, tensor_args={'device':'cpu','dtype':torch.float32}):
        super(MJRollout, self).__init__()
        self.dynamics_model = dynamics_model
        self.tensor_args = tensor_args

    def rollout_fn(self, start_state, mean_act, delta):
        trajectories = self.dynamics_model.rollout_open_loop(start_state, mean_act, delta)
        return trajectories

    def __call__(self, start_state, mean_act, delta):
        return self.rollout_fn(start_state, mean_act, delta)