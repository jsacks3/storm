import argparse
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from isaacgym import gymapi
from isaacgym import gymutil
from storm_kit.gym.core import Gym

import numpy as np
import torch
from tqdm import tqdm
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from storm_kit.util_file import get_gym_configs_path, load_yaml, join_path, get_assets_path
#from storm_kit.mpc.task.cartpole_task import CartpoleTask
from storm_kit.gym.cartpole_wrapper import CartpoleWrapper
from storm_kit.mpc.task.cartpole_isaac_task import CartpoleIsaacTask

def mpc_cartpole(args):
    device = 'cuda' if args.cuda else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}
    gym_file = 'cartpole-isaac.yml'
    task_file = 'cartpole-isaac.yml'

    sim_engine_params = load_yaml(join_path(get_gym_configs_path(), gym_file))
    sim_engine_params['headless'] = args.headless
    gym_instance = Gym(**sim_engine_params, create_env=False)
    gym = gym_instance.gym
    sim = gym_instance.sim
    viewer = gym_instance.viewer

    sim_params = sim_engine_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    env = CartpoleWrapper(gym_instance=gym_instance, **sim_params, tensor_args=tensor_args)
    mpc_controller = CartpoleIsaacTask(env=env, task_file=task_file, tensor_args=tensor_args)

    state = env.get_state().view(-1)
    state[2] = np.pi/2
    env.set_state(state)

    for i in tqdm(range(200)):
        state = env.get_state().view(-1)
        command = mpc_controller.get_command(state)
        env.set_state(state)
        env.set_control(command)
        gym_instance.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda.')
    parser.add_argument('--headless', action='store_true', default=False, help='Headless gym.')
    args = parser.parse_args()

    mpc_cartpole(args)