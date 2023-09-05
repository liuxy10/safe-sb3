from stable_baselines3 import IQL
import gym

import h5py
import os
from datetime import datetime
import numpy as np
# from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy

from utils import AddCostToRewardEnv
from stable_baselines3.common.callbacks import CheckpointCallback



WAYMO_SAMPLING_FREQ = 10




def main(args, is_test = False):
    env_name = "waymo"
    lamb = args["lambda"]
    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        num_scenarios = int(args['num_of_scenarios'])

    print("num of scenarios: ", num_scenarios)
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy": PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "start_seed": 0,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": False,
               "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    },lamb = lamb
    )
   
    env.seed(args["env_seed"])
    root_dir = "tensorboard_logs"
    experiment_name = (
        "iql-" + env_name + "_es" + str(args["env_seed"]) 
        # + "_lam" + str(lamb) + '_' + date)
        + "_lam" + str(lamb))
    tensorboard_log = os.path.join(root_dir, experiment_name)

    model = IQL("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1, device="cpu")
    model.learn(total_timesteps=args["steps"])
    model.save("iql-" + env_name + "-es" + str(args["env_seed"]))

    del model
    env.close()


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='/home/xinyi/src/data/metadrive/pkl_9/')
    
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=str, default="10000")
    parser.add_argument('--steps', '-st', type=int, default=int(100000))
    args = parser.parse_args()
    args = vars(args)

    main(args)





   


  
  


