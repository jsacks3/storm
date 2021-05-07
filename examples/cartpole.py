import copy
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import numpy as np

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_cartpole import CartpoleSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.mpc.model.simulated_model import SimulatedModel
from storm_kit.mpc.rollout.rollout_cartpole import CartpoleRollout
from storm_kit.mpc.task.cartpole_task import CartpoleTask

def mpc_cartpole(args, gym_instance, robot_params):
    gym = gym_instance.gym
    sim = gym_instance.sim
    viewer = gym_instance.viewer
    task_file = 'cartpole.yml'

    # Get simulation parameters
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    sim_params['collision_model'] = None

    # Set device
    if(args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Create the robot simulation
    robot_sim = CartpoleSim(gym_instance=gym_instance, **sim_params, device=device)
    robot_ptrs = robot_sim.spawn_robots()
    model = SimulatedModel(gym_instance=gym_instance, robot_sim=robot_sim, robot_ptrs=robot_ptrs, device=device,
                           tensor_args=tensor_args, d_obs=4, d_action=1)

    mpc_control = CartpoleTask(dynamics_model=model, task_file=task_file, tensor_args=tensor_args)
    sim_dt = mpc_control.exp_params['control_dt']
    t_step = gym_instance.get_sim_time()

    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_ptrs[0]

    # while True:
    #     #gym_instance.step()
    #     #     gym.simulate(sim)
    #     gym.fetch_results(sim, True)
    #     gym.step_graphics(sim)
    #     gym.draw_viewer(viewer, sim, True)
    #     gym.sync_frame_time(sim)
    #
    #     t_step += sim_dt
    #     current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    #     #command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=False)
    #
    #     #robot_sim.set_robot_state(current_robot_state['position'],
    #     #                          current_robot_state['velocity'],
    #     #                          env_ptr,
    #     #                          robot_ptr)
    #
    #     actions_tensor = torch.zeros(model.num_envs*model.num_dof, **tensor_args)
    #     actions_tensor[::model.num_dof] = torch.rand(model.num_envs, **tensor_args)
    #     #actions_tensor[::model.num_dof] = command.to(**tensor_args)
    #     efforts = gymtorch.unwrap_tensor(actions_tensor)
    #     gym.set_dof_actuation_force_tensor(sim, efforts)


    while True:
        t_step += sim_dt

        gym.refresh_dof_state_tensor(sim)
        pos = model.dof_pos[0].clone()
        vel = model.dof_vel[0].clone()
        current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))

        command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=False)

        #robot_sim.set_robot_state(current_robot_state['position'],
        #                         current_robot_state['velocity'],
        #                         env_ptr,
        #                         robot_ptr)

        gym.refresh_dof_state_tensor(sim)
        model.dof_pos[0, :] = pos
        model.dof_vel[0, :] = vel
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(model.dof_state))

        actions_tensor = torch.zeros(model.num_envs*model.num_dof, **tensor_args)
        actions_tensor[0] = command.to(**tensor_args)
        efforts = gymtorch.unwrap_tensor(actions_tensor)
        gym.set_dof_actuation_force_tensor(sim, efforts)

        gym.simulate(sim)

        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        #gym.sync_frame_time(sim)

if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use cuda.')
    parser.add_argument('--headless', action='store_true', default=False, help='Headless gym.')
    args = parser.parse_args()

    sim_params = load_yaml(join_path(get_gym_configs_path(), 'cartpole.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)

    mpc_cartpole(args, gym_instance, sim_params)
