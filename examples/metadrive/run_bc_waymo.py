
import gym

import h5py
import os
from datetime import datetime
import numpy as np
# from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy
from stable_baselines3 import BC
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from utils import AddCostToRewardEnv
import matplotlib.pyplot as plt

from visualize import plot_waymo_vs_pred

from stable_baselines3.common.callbacks import CheckpointCallback



WAYMO_SAMPLING_FREQ = 10




def main(args):

    
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
        "agent_policy":ReplayEgoCarPolicy, # BC uses ReplayEgoCarPolicy to train policy
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": False,
                "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
    }, 
    )
    env.seed(args["env_seed"])

    exp_name = "bc-waymo-es" + str(args["env_seed"])
    root_dir = "tensorboard_log"
    tensorboard_log = os.path.join(root_dir, exp_name)

    model = BC("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1)
   # Save a checkpoint every given steps
    # checkpoint_callback = CheckpointCallback(save_freq=args['save_freq'], save_path=args['output_dir'],
    #                                      name_prefix=exp_name)
    
    model.learn(
                args['steps'], 
                data_dir = args['h5py_path'], 
                # callback=checkpoint_callback, 
                use_diff_action_space = args['use_diff_action_space']
                )

    
    del model
    env.close()

def test(args):
    from collect_h5py_from_pkl import get_current_ego_trajectory_old

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
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": True,
                "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
    },
    )

    env.seed(args["env_seed"])
    
    model_dir = args["policy_load_dir"]
    model = BC("MlpPolicy", env)
    model.set_parameters(model_dir)

    for seed in range(0, num_scenarios):
        plot_waymo_vs_pred(env, model, seed, 'bc', savefig_dir = "examples/metadrive/figs/bc_vs_waymo/diff_action")
      
    del model
    env.close()






if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5py_path', '-h5', type=str, default='examples/metadrive/h5py/bc_9_900.h5py')
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_9')
    parser.add_argument('--output_dir', '-out', type=str, default='examples/metadrive/saved_bc_policy')
    parser.add_argument('--policy_load_dir', type=str, default = 'examples/metadrive/example_policy/bc-diff-peak.pt')
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=str, default="10")
    parser.add_argument('--steps', '-st', type=int, default=int(100000))
    parser.add_argument('--save_freq', type=int, default=int(10000))
    args = parser.parse_args()
    args = vars(args)

    main(args)
    # test(args)




   


  
  


