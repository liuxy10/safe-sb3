
import gym

import h5py
import os
from datetime import datetime
import numpy as np
# from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy, EnvInputPolicy
from stable_baselines3 import BC
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from visualize import plot_waymo_vs_pred
from utils import AddCostToRewardEnv
import matplotlib.pyplot as plt

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
        # "agent_policy":EnvInputHeadingAccPolicy,
        "agent_policy": PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": True,
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

    exp_name = "sac-waymo-es" + str(args["env_seed"])
    root_dir = "tensorboard_log"
    tensorboard_log = os.path.join(root_dir, exp_name)
    
    model = SAC("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1, buffer_size = 100000)
    # model = PPO("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1)
    # Save a checkpoint every given steps
    checkpoint_callback = CheckpointCallback(save_freq=args['save_freq'], save_path=args['output_dir'],
                                         name_prefix=exp_name)
    
    model.learn(args['steps'], callback=checkpoint_callback)
        
    # loaded_agent =PPO.load(exp_name)

    
    del model
    env.close()


def test(args):

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
        lidar=dict(num_lasers=120, distance=50, num_others=4),
        lane_line_detector=dict(num_lasers=12, distance=50),
        side_detector=dict(num_lasers=160, distance=50)
        ),
    },
    )

    env.seed(args["env_seed"])
    
    model_dir = args["policy_load_dir"]

    model = SAC("MlpPolicy", env)
    model.set_parameters(model_dir)

    avg_cost, avg_rew = 0,0
    for seed in range(0, num_scenarios):
        rew, cost = plot_waymo_vs_pred(env, model, seed, 'sac', savefig_dir = "examples/metadrive/figs/sac_vs_waymo/diff_action/test_20/")
        avg_rew +=  rew
        avg_cost +=  cost
        print("seed,  rew, cost = ", seed, rew, cost)
    avg_cost/= num_scenarios
    avg_rew/=num_scenarios

    print("avg. rew, cost out of "+str(num_scenarios)+" scenarios = ",avg_rew, avg_cost )
    del model
    env.close()





if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_20')
    parser.add_argument('--output_dir', '-out', type=str, default='examples/metadrive/saved_sac_policy')
    parser.add_argument('--policy_load_dir', type=str, default = 'examples/metadrive/example_policy/sac-diff-peak.pt')
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=str, default="10")
    parser.add_argument('--steps', '-st', type=int, default=int(100000))
    parser.add_argument('--save_freq', type=int, default=int(10000))
    args = parser.parse_args()
    args = vars(args)

    # main(args)
    test(args)