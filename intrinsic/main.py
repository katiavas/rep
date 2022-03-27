import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
import torch as T
import numpy as np
import random
import gym
# import wandb
from memory import Memory


os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['WANDB_START_METHOD'] = 'thread'

# wandb.init(project='icm', entity="katiavas", dir='./')

if __name__ == '__main__':
    SEED = 111
    random.seed(SEED)
    np.random.seed(SEED)
    T.manual_seed(SEED)
    mp.set_start_method('spawn', force=True)
    global_ep = mp.Value('i', 0)
    # env_id = 'PongNoFrameskip-v4'
    # env_id = 'MiniWorld-FourRooms-v0'
    env1 = 'ALE/Breakout-v5'
    env = gym.make(env1)
    env.seed(111)
    # env_id = 'CartPole-v1'
    n_threads = 12
    # n_actions = 4
    n_actions = 4
    input_shape = [4, 42, 42]
    ICM = True
    # wandb.run.name = env_id+'/'+str(SEED) + '/ICM='+str(ICM)
    env = ParallelEnv(env_id=env, num_threads=n_threads,
                      n_actions=n_actions, global_idx=global_ep,
                      input_shape=input_shape, icm=ICM)

# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
'''the state-space of the Cart-Pole has four dimensions of continuous values 
and the action-space has one dimension of two discrete values'''
