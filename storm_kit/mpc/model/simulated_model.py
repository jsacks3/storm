from .model_base import DynamicsModelBase
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import numpy as np
import torch

class SimulatedModel(DynamicsModelBase):
    def __init__(self, gym_instance, robot_sim, robot_ptrs, d_obs, d_action,
                 tensor_args={'device':'cpu','dtype':torch.float32}, device='cpu'):
        super(SimulatedModel, self).__init__()
        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        self.robot_sim = robot_sim
        self.device = device
        self.robot_ptrs = robot_ptrs
        self.tensor_args = tensor_args

        self.num_dof = self.robot_sim.dof
        self.num_envs = len(self.gym_instance.env_list)
        self.d_obs = d_obs
        self.d_action = d_action

        # Allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.d_obs), device=self.device, dtype=torch.float)

        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    def _set_states(self, pos, vel, env_ids=None):
        if env_ids is None:
            env_ids = torch.from_numpy(np.arange(self.num_envs)).to(dtype=torch.long, device=pos.device)

        self.dof_pos[env_ids, :] = pos[:, :]
        self.dof_vel[env_ids, :] = vel[:, :]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

    def rollout_open_loop(self, start_state, act_seq):
        inp_device = start_state.device
        start_state = start_state.to(**self.tensor_args)
        act_seq = act_seq.to(**self.tensor_args)

        pos = start_state[:, :self.num_dof].repeat(self.num_envs, 1)
        vel = start_state[:, self.num_dof:self.num_dof*2].repeat(self.num_envs, 1)
        self._set_states(pos, vel)

        batch_size, T, n_act = act_seq.shape
        n_state = start_state.shape[-1]
        states = torch.zeros((batch_size, T, n_state), device=self.device, dtype=start_state.dtype)

        for t in range(T):
            self.step(act_seq[:, t])
            states[:, t, :] = self.obs_buf.clone()

        state_dict = {'state_seq':states.to(inp_device)}
        return state_dict

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

    def get_next_state(self, curr_state, act, dt):
        pass

    def step(self, actions):
        self.pre_physics_step(actions)

        self.gym.simulate(self.sim)
        #self.gym.fetch_results(self.sim, True)
        #self.gym.step_graphics(self.sim)
        #self.gym.draw_viewer(self.gym_instance.viewer, self.sim, True)

        self.post_physics_step()

    def pre_physics_step(self, actions):
        # TODO: Move this to separate Cartpole class
        actions_tensor = torch.zeros(self.num_envs*self.num_dof, device=self.device, dtype=actions.dtype)
        actions_tensor[::self.num_dof] = actions.squeeze()
        efforts = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, efforts)

    def post_physics_step(self):
        self.compute_observations()