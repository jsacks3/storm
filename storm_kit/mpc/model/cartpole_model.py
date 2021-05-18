from .model_base import DynamicsModelBase
import numpy as np
import torch
import gym

class CartpoleModel(DynamicsModelBase):
    def __init__(self, tensor_args={'device':'cpu','dtype':torch.float32}):
        super(CartpoleModel, self).__init__()
        self.tensor_args = tensor_args

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.d_action = 1
        self.d_obs = 4
        self.action_lows = -self.force_mag * torch.ones(self.d_action, **self.tensor_args)
        self.action_highs = self.force_mag * torch.ones(self.d_action, **self.tensor_args)

    def step(self, state, action):
        x, x_dot, theta, theta_dot = state[:, :1], state[:, 1:2], state[:, 2:3], state[:, 3:]
        force = action
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        state = torch.cat((x, x_dot, theta, theta_dot), dim=-1)
        return state

    def rollout_open_loop(self, start_state, act_seq):
        num_particles, horizon, _ = act_seq.shape
        state_seq = torch.zeros((num_particles, horizon, self.d_obs), **self.tensor_args)
        state_t = start_state.unsqueeze(0).repeat((num_particles,1))

        for t in range(horizon):
            state_t = self.step(state_t, act_seq[:,t])
            state_seq[:, t] = state_t

        trajectories = dict(
            actions = act_seq,
            state_seq = state_seq
        )

        return trajectories

    def get_next_state(self, curr_state, act, dt):
        pass