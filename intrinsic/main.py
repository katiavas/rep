import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
import torch
from memory import Memory


os.environ['OMP_NUM_THREADS'] = '1'


if __name__ == '__main__':
    torch.manual_seed(5)
    mp.set_start_method('spawn')
    global_ep = mp.Value('i', 0)
    # env_id = 'PongNoFrameskip-v4'
    # env_id = 'MiniWorld-Hallway-v0'
    # env_id = 'MiniWorld-FourRooms-v0'
    env_id = 'ALE/Breakout-v5'
    n_threads = 12
    n_actions = 4
    input_shape = [4, 42, 42]
    env = ParallelEnv(env_id=env_id, num_threads=n_threads,
                      n_actions=n_actions, global_idx=global_ep,
                      input_shape=input_shape, icm=False)
                      
# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
'''the state-space of the Cart-Pole has four dimensions of continuous values 
and the action-space has one dimension of two discrete values'''
