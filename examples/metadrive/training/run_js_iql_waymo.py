import os
from datetime import datetime

import metadrive
import gym


import h5py
import os
from datetime import datetime
import numpy as np
# from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy, EnvInputPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import JumpStartIQL
from stable_baselines3.js_sac import utils as js_utils



from utils import AddCostToRewardEnv
import matplotlib.pyplot as plt
WAYMO_SAMPLING_FREQ = 10


def main(args):
    device = args["device"]
    lamb = args["lambda"]
    use_transformer_expert = args["use_transformer_expert"]
    
    print("use_transformer_expert",use_transformer_expert)
    
    # import pdb; pdb.set_trace()

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
        "start_seed": 0,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "horizon": 90/5,
        "reactive_traffic": False,
                 "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    }, lamb= lamb
    )
    env.seed(args["env_seed"])
    if args["random"]:
        env.set_num_different_layouts(100)

    # specify tensorboard log settings
    root_dir = "tensorboard_logs"
    experiment_name = (
        "js-iql-waymo_es" + str(args["env_seed"])
        + "_lamb" + str(lamb))
    if args["suffix"]:
        experiment_name += f'_{args["suffix"]}'
    if use_transformer_expert:
        experiment_name += '_transformer'
    tensorboard_log = os.path.join(root_dir, experiment_name)
    # print("use_transformer_expert",use_transformer_expert)
    if use_transformer_expert:
        loaded_stats = js_utils.load_demo_stats(
            path=args["expert_model_dir"]
        )
        obs_mean, obs_std, reward_scale, target_return = loaded_stats
        expert_policy = js_utils.load_transformer(
            model_dir=args['expert_model_dir'], device=device
        )

        ## TODO: delete this when updated model is loaded :
        if reward_scale == None:
            reward_scale, target_return = 100, 400
    

    

    else:
        obs_mean, obs_std = None, None
        expert_policy = js_utils.load_expert_policy(
            model_dir=args['expert_model_dir'], env=env, device=device
        )

    model = JumpStartIQL(
        "MlpPolicy",
        expert_policy,
        env,
        use_transformer_expert=use_transformer_expert,
        target_return=target_return,
        reward_scale=reward_scale,
        obs_mean=obs_mean,
        obs_std=obs_std,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
    )
    
    total_timesteps=args["steps"]
    num_chunks = args["num_chunks"]
    step_per_chunk = total_timesteps/num_chunks
    last_timestep = 0
    for i in range(num_chunks):
        
        print("-"*100)
        print(f"restarting from {i+1} / {num_chunks}, timestep {last_timestep}" )
        
        if i == 0:
            model.learn(total_timesteps=step_per_chunk, 
                        reset_num_timesteps=False)
            model_dir = model.logger.dir
            last_timestep = model.num_timesteps
            del model

        else:
            model = JumpStartIQL(
                "MlpPolicy",
                expert_policy,
                env,
                use_transformer_expert=use_transformer_expert,
                target_return=target_return,
                reward_scale=reward_scale,
                obs_mean=obs_mean,
                obs_std=obs_std,
                tensorboard_log=tensorboard_log,
                verbose=1,
                device=device,
             )
            model.set_parameters(os.path.join(model_dir, 'model.pt'))
            model.num_timesteps = last_timestep + 1
            model.learn(total_timesteps=step_per_chunk, 
                        reset_num_timesteps=False)
            
            last_timestep = model.num_timesteps
            del model

    
    
    env.close()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='/home/xinyi/src/data/metadrive/pkl_9')
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--device', '-d', type=str, default="cuda")
    parser.add_argument('--expert_model_dir', '-emd', type=str, default='/home/xinyi/src/decision-transformer/gym/wandb/run-20230822_180622-20swd1g8/')
    parser.add_argument('--use_transformer_expert',  type=bool, default=True)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=int, default=100)# 10

    parser.add_argument('--steps', '-st', type=int, default=int(1e5)) #1e6
    parser.add_argument('--num_chunks', type=int, default=500)
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--suffix', type=str)
    args = parser.parse_args()
    args = vars(args)



    main(args)