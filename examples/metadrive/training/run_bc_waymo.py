
import gym

import h5py
import os
from datetime import datetime
import numpy as np
# from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy
from stable_baselines3 import BC
from stable_baselines3.common.evaluation import evaluate_policy
from utils import AddCostToRewardEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# from visualize import plot_waymo_vs_pred

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback



WAYMO_SAMPLING_FREQ = 10




def main(args):

    
    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        num_scenarios = int(args['num_of_scenarios'])
    # # TODO: delete this when test done
    # num_scenarios = 100

    print("num of scenarios: ", num_scenarios)
    env = AddCostToRewardEnv(
    {
            "manual_control": False,
            "no_traffic": False,
            "agent_policy": PMKinematicsEgoPolicy,
            "start_seed": 0,
            "waymo_data_directory": args['pkl_dir'],
            "case_num": num_scenarios,
            # have to be specified each time we use waymo environment for training purpose
            "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ,
            "use_render": False,
            "horizon": 90/5,
            "reactive_traffic": False,
            "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=80, distance=50, num_others=4),  # 120
                lane_line_detector=dict(num_lasers=12, distance=50),  # 12
                side_detector=dict(num_lasers=20, distance=50))  # 160,
        },
    )
   
    env.seed(args["env_seed"])

    exp_name = "bc-waymo-cost-default"
    root_dir = "tensorboard_log"
    tensorboard_log = os.path.join(root_dir, exp_name)
    num_chunks = 50
    step_per_chunk = 2e4 #TODO: change back to 2e4
    print("step_per_chunk, first round = ", step_per_chunk, args['first_round'])
    last_timestep = 0
    env_config = env.config
    buffer_path = "/home/xinyi/src/safe-sb3/examples/metadrive/training/bc_replay_buffer.h5py"
    params_path = "/home/xinyi/src/safe-sb3/examples/metadrive/training/bc_params.npy"
    model_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0"

    if args['first_round']:

        model = BC(
        "MlpPolicy", 
        env, 
        tensorboard_log=tensorboard_log,
        verbose=1
        )
        
        model.learn(total_timesteps=step_per_chunk,
                    data_dir = args['h5py_path'], 
                    use_diff_action_space = args['use_diff_action_space'],
                    reset_num_timesteps=False)
        last_timestep = model.num_timesteps
        model.save_replay_buffer(buffer_path)
        np.save(params_path,last_timestep)
        model.save(os.path.join(model.logger.dir, "last_model.pt"))
    
        del model
        env.close()
        del env

    else:
        env = AddCostToRewardEnv(env_config)
        
        model = BC.load(os.path.join(model_dir, 'last_model.pt'),
                        env,
                        print_system_info= True)
    
        model.reload_replay_buffer(buffer_path)
        last_timestep = np.load(params_path)
        model.num_timesteps = last_timestep + 1

        model.learn(total_timesteps=step_per_chunk,
                    data_dir = args['h5py_path'], 
                    use_diff_action_space = args['use_diff_action_space'],
                    reset_num_timesteps=False)
     
        model.save_replay_buffer(buffer_path)
        model.save(os.path.join(model.logger.dir, "last_model.pt"))
        last_timestep = model.num_timesteps
        np.save(params_path,last_timestep)
        del model
        env.close()
        del env

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_round','-f', action="store_true")
    parser.add_argument('--h5py_path', '-h5', type=str, default='examples/metadrive/h5py/bc_9_1000000.h5py')
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_9')
    
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--num_of_scenarios', type=str, default="100")
    parser.add_argument('--steps', '-st', type=int, default=int(100000))
    parser.add_argument('--num_chunks', type=int, default=50)
    args = parser.parse_args()
    args = vars(args)

    main(args)





   


  
  


