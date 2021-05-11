import copy
import numpy as np
import torch

try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
except Exception:
    print("ERROR: gym not loaded, this is okay when generating doc")

from .helpers import load_struct_from_dict
from ..util_file import join_path

class GymWrapper():
    def __init__(self, gym_instance, asset_root='', sim_urdf='', asset_options='', init_state=None,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}, **kwargs):
        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        self.viewer = gym_instance.viewer
        self.tensor_args = tensor_args
        self.init_state = init_state
        self.device = tensor_args['device']

        self.num_envs = self.gym_instance.num_envs
        self.env_list = []
        self.asset_ptrs = []

        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options = load_struct_from_dict(robot_asset_options, asset_options)
        self.robot_asset = self.load_robot_asset(sim_urdf,
                                                 robot_asset_options,
                                                 asset_root)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)

        self.create_envs(self.num_envs, num_per_row=int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)

        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    def set_state(self, state, env_ids=torch.tensor([0])):
        if env_ids is None:
            env_ids = torch.from_numpy(np.arange(self.num_envs))
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        state = state.view(-1, self.num_dof, 2)
        dof_state = self.dof_state.view(self.num_envs, self.num_dof, 2)
        dof_state[env_ids, :, :] = state

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

    def get_state(self, env_ids=torch.tensor([0])):
        if env_ids is None:
            env_ids = torch.from_numpy(np.arange(self.num_envs))
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        self.gym.refresh_dof_state_tensor(self.sim)
        return self.dof_state.view(self.num_envs, self.num_dof, 2)[env_ids]

    def set_control(self, action, env_ids=torch.tensor([0])):
        if env_ids is None:
            env_ids = torch.from_numpy(np.arange(self.num_envs))
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        action.to(**self.tensor_args)
        env_ids_int32 = gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32))
        efforts = gymtorch.unwrap_tensor(action)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        efforts,
                                                        env_ids_int32,
                                                        env_ids_int32.shape[0])

    def load_robot_asset(self, sim_urdf, asset_options, asset_root):
        if ((self.gym is None) or (self.sim is None)):
            raise AssertionError
        robot_asset = self.gym.load_asset(self.sim, asset_root,
                                          sim_urdf, asset_options)
        return robot_asset

    def create_envs(self, num_envs, spacing=1.0, num_per_row=1):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        robot_pose = self.init_state
        p = gymapi.Vec3(robot_pose[0], robot_pose[1], robot_pose[2])
        robot_pose = gymapi.Transform(p=p, r=gymapi.Quat(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6]))


        for i in range(num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.env_list.append(env_ptr)

            asset_ptr = self.gym.create_actor(env_ptr, self.robot_asset, robot_pose, 'robot', i, i, 0)
            self.asset_ptrs.append(asset_ptr)
            self._init_assets(env_ptr, asset_ptr)

    def _init_assets(self, env_ptr, asset_ptr):
        pass
