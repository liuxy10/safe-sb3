from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
import json

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import JumpStartIQL, BC, JumpStartSAC, SAC, IQL
from stable_baselines3.iql.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy, ValueNet
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.js_sac import utils as js_utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic

from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
import tqdm
import sys

sys.path.append("examples/metadrive/training")
from visualize import plot_waymo_vs_pred
from utils import AddCostToRewardEnv



def evaluate_model_under_env(
        training_method, 
        env_test, 
        policy_load_dir = "",  
        save_fig_dir = "",
        model_config = {}, 
        start_seed = 10000, 
        episode_len = 90,
        end_eps_when_done = True
        ):


    if training_method in (BC, SAC, IQL):
        # model = training_method("MlpPolicy", env_test)

        # model.set_parameters(policy_load_dir)

        model = training_method.load(policy_load_dir, 
                            env_test,
                            device = 'cpu'
                            )
        fn = training_method.__name__ 
        

    elif training_method in (JumpStartIQL, JumpStartSAC):
        # should be able to load all useful info from current env.
        
        
        model = training_method.load(policy_load_dir, 
                            env_test,
                            device = 'cpu',
                            kwargs= model_config
                            )

        fn = training_method.__name__ +"_dt=" + str(model_config['use_transformer_expert'])
        
    

    else:
        print("[eval] method not implememented!")
        return 

    header = "-"*10+" Evaluation of " + fn + "-"*10
    mean_reward, std_reward,  mean_cost, std_cost, mean_success_rate= evaluate_policy(model, 
                                                                env_test, 
                                                                n_eval_episodes=env_test.config['case_num'], 
                                                                deterministic=True, 
                                                                render=True)
    
    print(header)
    print("mean_reward = ", mean_reward)
    print("std_reward = ",std_reward)
    print("mean_cost = ", mean_cost)
    print("std_cots = ",std_cost)
    print("mean_success_rate = ", mean_success_rate)

    exp_result ={
        "exp_name": fn,
        "policy_load_dir": policy_load_dir,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "mean_success_rate": mean_success_rate
    }
    json_path = fn + ".json"

    # Save the dictionary as a JSON config file
    with open(json_path, "w") as json_file:
        json.dump(exp_result, json_file, indent=4)



    for seed in tqdm.tqdm(range( start_seed,  start_seed + env_test.config['case_num'])):
        plot_waymo_vs_pred(env_test, 
                           model, 
                           seed, 
                           training_method.__name__, 
                           savefig_dir = os.path.join(save_fig_dir, fn), 
                           end_eps_when_done = end_eps_when_done)

        # print("mean_reward, std_reward, mean_success_rate = ", mean_reward, std_reward, mean_success_rate )


def all_elements_in_dict_keys(elements, dictionary):
    # print(dictionary)
    for element in elements:
        if element not in dictionary:
            return False
    return True

def main(args):
    env_config =  {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": 100,
        "start_seed": 10000, 
        "physics_world_step_size": 1/10, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "horizon": 90/5,
        "reactive_traffic": False,
                 "vehicle_config": dict(
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    }
  
    env = AddCostToRewardEnv(env_config)

    # BC only 
    if args['algorithm'] == 'bc':
        print("test bc only")
        evaluate_model_under_env(BC, env, 
            policy_load_dir = '/home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0/model.pt',
            # policy_load_dir= '/home/xinyi/src/safe-sb3/tensorboard_log/bc-waymo-cost-default/BC_1000/model.pt',
            # policy_load_dir= '/home/xinyi/src/safe-sb3/tensorboard_log/bc-waymo-es0/BC_0/model.pt',
            save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/",
            start_seed= 10000
            )
        env.close()


     # iql only 
    if args['algorithm'] == 'iql':
        print("test iql only")
        evaluate_model_under_env(IQL, env, 
            # policy_load_dir = '/home/xinyi/src/safe-sb3/examples/metadrive/results/tb/iql-waymo_es0_lam1.0/msc/model.pt',
            # policy_load_dir= '/home/xinyi/src/safe-sb3/examples/metadrive/results/tb/iql-waymo_es0_lam1.0/IQL_2/last_model.pt',
            policy_load_dir= '/home/xinyi/src/safe-sb3/results/tb/iql/1/model.pt',
            save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/",
            start_seed= 10000
            )
        env.close()
    
    

    ## test of DT still goes into the eval in DT repo
   
   # js-dt
    elif args['algorithm'] == 'dt-js-iql':
        
        # JS-iql policy 
        # policy_load_dir = '/home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_logs/js-iql-waymo_es0_lamb1.0_transformer/IQL_0/model.pt'   
        policy_load_dir = '/home/xinyi/src/safe-sb3/results/tb/dt-js-iql/1/last_model.pt'   
        # DT policy as expert policy
        expert_policy_dir = '/home/xinyi/src/decision-transformer/gym/wandb/run-20230823_230743-3s6y7mzy' # acc bounded 
        
        expert_policy = js_utils.load_transformer(expert_policy_dir, device='cpu')
        loaded_stats = js_utils.load_demo_stats(
                path=expert_policy_dir
            )
        obs_mean, obs_std, reward_scale, target_return = loaded_stats
        model_config = {
                'expert_policy': expert_policy,
                'use_transformer_expert': True,
                'target_return': target_return,
                'reward_scale':reward_scale, 
                'obs_mean':obs_mean, 
                'obs_std':obs_std,
                'verbose':1,
                'tensorboard_log':''
                }
    
        evaluate_model_under_env(JumpStartIQL, env, 
            policy_load_dir = policy_load_dir,
            save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/",
            model_config = model_config
            )
        env.close()
    
    elif args['algorithm'] == 'bc-js-iql':
        print("test JS-iql, with bc as guide policy ")
        # JS-iql policy 
        # policy_load_dir = '/home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_logs/js-iql-waymo_es0_lamb1.0/IQL_0/model.pt'   
        policy_load_dir ='/home/xinyi/src/safe-sb3/results/tb/bc-js-iql/1/last_model.pt'
        # BC policy as expert policy
        expert_policy_dir =  '/home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_log/bc-waymo-cost-default/BC_0/model.pt'
        expert_policy = js_utils.load_expert_policy(expert_policy_dir, env)
        model_config = {
                'model_dir': policy_load_dir,
                'expert_policy': expert_policy,
                'use_transformer_expert': False,
                'target_return': None,
                'reward_scale':None, 
                'obs_mean':None, 
                'obs_std':None,
                'verbose':1,
                'tensorboard_log':''
                }
    
        evaluate_model_under_env(JumpStartIQL, env, 
            policy_load_dir = policy_load_dir,
            save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/",
            model_config = model_config
            )
        env.close()




    


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='/home/xinyi/src/data/metadrive/pkl_9/')
    parser.add_argument('--algorithm', '-alg', type=str, default='iql')
    
    parser.add_argument('--policy_load_dir', type=str, default = '')
    parser.add_argument('--save_result_dir', type=str, default = 'eval_results')
    
    args = parser.parse_args()
    args = vars(args)

    main(args)
    