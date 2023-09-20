from stable_baselines3 import IQL
import gym
import json
import h5py
import os
from datetime import datetime
import numpy as np
# from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy

from utils import AddCostToRewardEnv



WAYMO_SAMPLING_FREQ = 10




def main(args):
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
    root_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/results/tb"
    experiment_name = (
        "iql-" + env_name + "_es" + str(args["env_seed"]) 
        # + "_lam" + str(lamb) + '_' + date)
        + "_lam" + str(lamb))
    tensorboard_log = os.path.join(root_dir, experiment_name)
    
    print("step_per_chunk, first round = ", args['steps'], args['first_round'])
    last_timestep = 0
    env_config = env.config
    

    if args['first_round']:
    
        model = IQL(
            "MlpPolicy", 
            env, 
            tensorboard_log=tensorboard_log, 
            verbose=1, 
            device="cpu")
        
        model.learn(total_timesteps=args['steps'])
        last_timestep = model.num_timesteps
        
        model.save(os.path.join(model.logger.dir, "last_model.pt"))
        buffer_path = os.path.join(model.logger.dir, "replay_buffer.pkl")
        model.save_replay_buffer(buffer_path)

        last_model_info = {
            "last_timestep": last_timestep,
            "model.logger.dir": model.logger.dir,
        }
        json_path = experiment_name + ".json"

        # Save the dictionary as a JSON config file
        with open(json_path, "w") as json_file:
            json.dump(last_model_info, json_file, indent=4)
        del model
        env.close()
        del env

    else:
        env = AddCostToRewardEnv(env_config)
        json_path = experiment_name + ".json"

        # Load the JSON config file as a dictionary
        with open(json_path, "r") as json_file:
            last_model_info = json.load(json_file)
        
        model = IQL.load(os.path.join(last_model_info['model.logger.dir'], 'last_model.pt'),
                                        env, 
                                        print_system_info= True,
                                        device="cpu")
        buffer_path = os.path.join(last_model_info['model.logger.dir'], "replay_buffer.pkl")
        model.load_replay_buffer(buffer_path)
        
        last_timestep = last_model_info["last_timestep"]
        model.num_timesteps = last_timestep + 1
        
        model.learn(total_timesteps=args['steps'],
                    reset_num_timesteps=False)

        model.save_replay_buffer(buffer_path)
        model.save(os.path.join(model.logger.dir, "last_model.pt"))
        last_timestep = model.num_timesteps

        last_model_info = {
            "last_timestep": last_timestep,
            "model.logger.dir": model.logger.dir,
        }
        json_path = experiment_name + ".json"


        # Save the dictionary as a JSON config file
        with open(json_path, "w") as json_file:
            json.dump(last_model_info, json_file, indent=4)
        
        del model
        env.close()
        del env

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='/home/xinyi/src/data/metadrive/pkl_9/')
    parser.add_argument('--first_round','-f', action="store_true")
    
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=int, default=100)
    parser.add_argument('--steps', '-st', type=int, default=100) # 100000
    args = parser.parse_args()
    args = vars(args)

    main(args)





   


  
  


