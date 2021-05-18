import argparse
import torch
import gym
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import mj_envs
from mjmpc.envs import GymEnvWrapper
from storm_kit.gym.cartpole import CartPoleEnv

from storm_kit.util_file import get_gym_configs_path, load_yaml, join_path, get_mpc_configs_path
from storm_kit.mpc.task.cartpole_analytic_task import CartpoleAnalyticTask

def mpc_cartpole(args):
    device = 'cuda' if args.cuda else 'cpu'
    tensor_args = {'device': device, 'dtype': torch.float32}
    task_file = 'cartpole-analytic.yml'

    # Create the main environment
    env = CartPoleEnv()
    env.reset()

    # Create the simulation model
    mpc_controller = CartpoleAnalyticTask(task_file=task_file, config_root=get_mpc_configs_path(),
                                          tensor_args=tensor_args)

    while True:
        curr_state = torch.tensor(copy.deepcopy(env.state), **tensor_args)
        command = mpc_controller.get_command(curr_state).cpu().numpy()
        obs, reward, done, info = env.step(command)
        env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use cuda.')
    parser.add_argument('--headless', action='store_true', default=False, help='Headless gym.')
    args = parser.parse_args()

    mpc_cartpole(args)