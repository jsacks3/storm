from .rollout_base import RolloutBase
import torch
import numpy as np

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class CartpoleRollout(RolloutBase):
    def __init__(self, dynamics_model, tensor_args={'device':'cpu','dtype':torch.float32}):
        super(CartpoleRollout, self).__init__()
        self.dynamics_model = dynamics_model
        self.tensor_args = tensor_args

    def cost_fn(self, state, act):
        slider_pos = state[:, :, 0]
        pole_ang = angle_normalize(state[:, :, 1])
        slide_vel = state[:, :, 2]
        pole_vel = state[:, :, 3]

        cost = slider_pos ** 2
        cost += pole_ang ** 2
        cost += 0.01 * slide_vel ** 2
        cost += 0.01 * pole_vel ** 2
        cost += 0.1 * act[:, :, 0] ** 2
        return cost

    def rollout_fn(self, start_state, mean_act, delta):
        trajectories = self.dynamics_model.rollout_open_loop(start_state, mean_act, delta)
        state_seq = trajectories['state_seq']
        act_seq = trajectories['actions']
        cost_seq = self.cost_fn(state_seq, act_seq)

        trajectories = dict(
            actions = act_seq,
            costs = cost_seq,
            rollout_time=0.0,
            state_seq = state_seq
        )

        return trajectories

    def __call__(self, start_state, mean_act, delta):
        return self.rollout_fn(start_state, mean_act, delta)