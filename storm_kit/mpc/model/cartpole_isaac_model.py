from .isaac_model import IsaacModel
import torch

class CartpoleIsaacModel(IsaacModel):
    def __init__(self, gym_instance, env, tensor_args={'device': 'cpu', 'dtype': torch.float32}):
        super(CartpoleIsaacModel, self).__init__(gym_instance, env, 4, 1, tensor_args)

        self.force_mag = 10.0
        self.action_lows = -self.force_mag * torch.ones(self.d_action, **self.tensor_args)
        self.action_highs = self.force_mag * torch.ones(self.d_action, **self.tensor_args)

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs*self.num_dof, **self.tensor_args)
        actions_tensor[::self.num_dof] = actions.squeeze()
        self.env.set_control(actions_tensor, env_ids=None)
