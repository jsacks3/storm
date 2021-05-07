from .rollout_base import RolloutBase
import torch
import numpy as np

class CartpoleRollout(RolloutBase):
    def __init__(self, dynamics_model, tensor_args={'device':'cpu','dtype':torch.float32}):
        super(CartpoleRollout, self).__init__()
        self.dynamics_model = dynamics_model
        self.tensor_args = tensor_args

    def _angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def cost_fn(self, state_dict, act_seq):
        state_seq = state_dict['state_seq']
        pole_angle = state_seq[:, :, 2]
        pole_vel = state_seq[:, :, 3]
        cart_vel = state_seq[:, :, 1]
        cart_pos = state_seq[:, :, 0]

        pole_angle = self._angle_normalize(pole_angle)
        cost = cart_pos*cart_pos
        cost += pole_angle*pole_angle
        cost += 0.01*pole_vel*pole_vel
        cost += 0.01*cart_vel*cart_vel
        cost += 0.1*act_seq[:, :, 0]*act_seq[:, :, 0]
        return -cost

    def rollout_fn(self, start_state, act_seq):
        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        cost_seq = self.cost_fn(state_dict, act_seq)

        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            state_seq=state_dict['state_seq'],
            rollout_time=0.0
        )
        return sim_trajs

    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)

    def current_cost(self, current_state):
        pass

    def update_params(self):
        pass