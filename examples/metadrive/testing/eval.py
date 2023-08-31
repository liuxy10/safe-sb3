from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import JumpStartIQL, BC, JumpStartSAC, SAC
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
        episode_len = 90
        ):


    if training_method in (BC, SAC):
        model = training_method("MlpPolicy", env_test)

        model.set_parameters(policy_load_dir)
        fn = training_method.__name__ 
        

    elif training_method in (JumpStartIQL, JumpStartSAC):
        # should be able to load all useful info from current env.
        # keys = ('expert_policy','use_transformer_expert', 'target_return', 'reward_scale', 'obs_mean', 'obs_std')
        # assert all_elements_in_dict_keys(keys, model_config), print('Model missing arguments, check keys') 
        # model = training_method(
        #     "MlpPolicy",
        #     env_test,
        #     model_config['expert_policy'],
        #     use_transformer_expert = model_config['use_transformer_expert'],
        #     target_return=model_config['target_return'],
        #     reward_scale=model_config['reward_scale'],
        #     obs_mean=model_config['obs_mean'],
        #     obs_std=model_config['obs_std'],
        #     device='cpu'
        # )
        model_dir = model_config['model_dir']
        model = training_method.load(model_dir, 
                            env_test,
                            kwargs= model_config
                            )

        
        fn = training_method.__name__ +"_dt=" + str(model_config['use_transformer_expert'])
        
    

    else:
        print("[eval] method not implememented!")
        return 

    header = "-"*10+" Evaluation of " + fn + "-"*10
    mean_reward, std_reward, mean_success_rate= evaluate_policy(model, env_test, n_eval_episodes=env_test.config['case_num'], deterministic=True, render=False)
    
    print(header)
    print("mean_reward = ", mean_reward)
    print("std_reward = ",std_reward)
    print("mean_success_rate = ", mean_success_rate)

    for seed in tqdm.tqdm(range( start_seed,  start_seed + env_test.config['case_num'])):
        plot_waymo_vs_pred(env_test, model, seed, training_method.__name__, savefig_dir = os.path.join(save_fig_dir, fn))

        # print("mean_reward, std_reward, mean_success_rate = ", mean_reward, std_reward, mean_success_rate )


def evaluate_guide_policy_only(env, use_transformer_expert, expert_model_dir, save_fig_dir):
    if use_transformer_expert:
        loaded_stats = js_utils.load_demo_stats(path=expert_model_dir)
        obs_mean, obs_std, reward_scale, target_return = loaded_stats
        expert_policy = js_utils.load_transformer(
            model_dir=expert_model_dir, device='cpu'
        )
        ## TODO: delete this when updated model is loaded :
        reward_scale, target_return = 100, 400
    
    else:
        obs_mean, obs_std = None, None
        expert_policy = js_utils.load_expert_policy(
            model_dir=expert_model_dir, env=env, device='cpu'
        )
        reward_scale, target_return = None, None
    
    model_config = {
        'expert_policy': expert_policy,
        'use_transformer_expert':  use_transformer_expert, 
        'target_return': target_return, 
        'reward_scale':reward_scale, 
        'obs_mean': obs_mean, 
        'obs_std': obs_std, 
    }
    evaluate_model_under_env(GuidePolicyOnly, env, model_config = model_config, save_fig_dir = save_fig_dir)
    
    env.close()


def all_elements_in_dict_keys(elements, dictionary):
    # print(dictionary)
    for element in elements:
        if element not in dictionary:
            return False
    return True


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='/home/xinyi/src/data/metadrive/pkl_9/')
    
    parser.add_argument('--policy_load_dir', type=str, default = '')
    
    args = parser.parse_args()
    args = vars(args)
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
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    }
  
    
    # test BC 
    # env = AddCostToRewardEnv(env_config)
    # evaluate_model_under_env(BC, env, 
    #     policy_load_dir = '/home/xinyi/src/safe-sb3/tensorboard_log/bc-waymo-cost-default/BC_1000/model.pt',
    #     save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/"
    #     )
    # env.close()
    
    # test SAC
    # env = AddCostToRewardEnv(env_config)
    # evaluate_model_under_env(SAC, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt',
        # save_fig_dir = save_fig_dir
        # )
    # env.close()
    
    # # test DT guide policy only:
    # env = AddCostToRewardEnv(env_config)
    # evaluate_guide_policy_only(env, use_transformer_expert = True, 
    #     expert_model_dir = "/home/xinyi/src/decision-transformer/gym/wandb/run-20230822_180622-20swd1g8",
    #     save_fig_dir = save_fig_dir
    #     )
    # env.close()

    # # test BC guide policy(redundant but interesting to try, should have the same result as 'test BC')
    # env = AddCostToRewardEnv(env_config)
    # evaluate_guide_policy_only(env, use_transformer_expert = False, 
    #     expert_model_dir = 'examples/metadrive/example_policy/bc-diff-peak-10000.pt',
    #     save_fig_dir = save_fig_dir
    #     )
    # env.close()

    # test JS-iql, with dt as guide policy 
    # env = AddCostToRewardEnv(env_config)
    # evaluate_model_under_env(JumpStartIQL, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt',
        # save_fig_dir = save_fig_dir
        # )
    # env.close()

    # test JS-iql, with bc as guide policy 
    env = AddCostToRewardEnv(env_config)
    
    expert_policy = js_utils.load_expert_policy('/home/xinyi/src/safe-sb3/tensorboard_log/bc-waymo-cost-default/BC_1000/model.pt', # BC policy 
                                                env)
    model_config = {
            'model_dir': '/home/xinyi/src/safe-sb3/examples/metadrive/training/tensorboard_logs/js-iql-waymo_es0_lamb1.0_transformer/IQL_0/model.pt',
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
        policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt',
        save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/",
        model_config = model_config
        )
    env.close()

