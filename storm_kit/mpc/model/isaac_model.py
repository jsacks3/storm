from .model_base import DynamicsModelBase
import numpy as np
import torch

try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
except Exception:
    print("ERROR: gym not loaded, this is okay when generating doc")

class IsaacModel(DynamicsModelBase):
    def __init__(self, gym_instance, env, d_obs, d_action, tensor_args={'device':'cpu','dtype':torch.float32}):
        super(IsaacModel, self).__init__()
        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        self.env = env
        self.tensor_args = tensor_args
        self.d_obs = d_obs
        self.d_action = d_action
        self.num_envs = env.num_envs
        self.num_dof = env.num_dof

    def step(self, actions):
        self.pre_physics_step(actions)
        self.gym.simulate(self.sim)

        #self.gym.fetch_results(self.sim, True)
        #self.gym.step_graphics(self.sim)
        #self.gym.draw_viewer(self.gym_instance.viewer, self.sim, True)

        self.post_physics_step()

    def rollout_open_loop(self, start_state, act_seq):
        num_particles, horizon, _ = act_seq.shape

        inp_device = start_state.device
        start_state = start_state.to(**self.tensor_args)
        act_seq = act_seq.to(**self.tensor_args)

        start_state = start_state.unsqueeze(0).repeat((self.env.num_envs, 1))
        self.env.set_state(start_state, env_ids=None)
        state_seq = torch.zeros((num_particles, horizon, self.d_obs), **self.tensor_args)

        for t in range(horizon):
            self.step(act_seq[:, t])
            state_t = self.env.get_state(env_ids=None)
            state_seq[:, t] = state_t.view(num_particles, -1)

        trajectories = dict(
            actions = act_seq,
            state_seq = state_seq.to(device=inp_device)
        )

        return trajectories

    def get_next_state(self, curr_state, act, dt):
        pass

    def pre_physics_step(self, actions):
        self.env.set_control(actions, env_ids=None)

    def post_physics_step(self):
        pass
