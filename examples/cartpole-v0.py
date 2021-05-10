import argparse
import torch
import gym
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import mj_envs
from mjmpc.envs import GymEnvWrapper

from storm_kit.mpc.model.mj_model import MJModel
from storm_kit.util_file import get_gym_configs_path, load_yaml, join_path, get_mpc_configs_path
from storm_kit.mpc.task.cartpole_task import CartpoleTask

def mpc_cartpole(args):
    device = 'cuda' if args.cuda else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}
    gym_file = 'cartpole-v0.yml'
    task_file = 'cartpole-v0.yml'

    # Create the main environment
    exp_params = load_yaml(join_path(get_gym_configs_path(), gym_file))
    env_name = exp_params['env_name']
    env = gym.make(env_name)
    env = GymEnvWrapper(env)
    env.real_env_step(True)

    # Create the simulation model
    mpc_controller = CartpoleTask(task_file=task_file, tensor_args=tensor_args)

    while True:
        curr_state = copy.deepcopy(env.get_env_state())
        command = mpc_controller.get_command(curr_state).cpu().numpy()
        obs, reward, done, info = env.step(command)
        env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use cuda.')
    parser.add_argument('--headless', action='store_true', default=False, help='Headless gym.')
    args = parser.parse_args()

    mpc_cartpole(args)