import numpy as np
import torch

try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
except Exception:
    print("ERROR: gym not loaded, this is okay when generating docs")

from .helpers import load_struct_from_dict
from .gym_wrapper import GymWrapper

class CartpoleWrapper(GymWrapper):
    def __init__(self, gym_instance, asset_root='', sim_urdf='', asset_options='', init_state=None,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}, **kwargs):
        super(CartpoleWrapper, self).__init__(gym_instance, asset_root, sim_urdf, asset_options,
                                              init_state, tensor_args, **kwargs)

        cam_pos = gymapi.Vec3(0, 10, 10)
        cam_target = gymapi.Vec3(0, 2, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _init_assets(self, env_ptr, asset_ptr):
        dof_props = self.gym.get_actor_dof_properties(env_ptr, asset_ptr)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0
        self.gym.set_actor_dof_properties(env_ptr, asset_ptr, dof_props)

        #self.gym.set_rigid_body_color(env_ptr, asset_ptr, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.6, 0.2))
        #self.gym.set_rigid_body_color(env_ptr, asset_ptr, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.5, 0.7))
        #self.gym.set_rigid_body_color(env_ptr, asset_ptr, 2, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.3))

    def set_control(self, action, env_ids=torch.tensor([0])):
        if env_ids is None:
            env_ids = torch.from_numpy(np.arange(self.num_envs))
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        action = action.to(**self.tensor_args)
        if action.shape[0] != env_ids.shape[0]*self.num_dof:
            actions_tensor = torch.zeros(env_ids.shape[0]*self.num_dof, **self.tensor_args)
            actions_tensor[::self.num_dof] = action.squeeze()
        else:
            actions_tensor = action

        env_ids_int32 = gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32))
        efforts = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        efforts,
                                                        env_ids_int32,
                                                        env_ids_int32.shape[0])
