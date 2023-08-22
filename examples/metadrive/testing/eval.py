from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import JumpStartIQL, BC, JumpStartSAC, SAC
from stable_baselines3.iql.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy, ValueNet
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.js_sac import utils as js_utils

from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy

import sys

sys.path.append("examples/metadrive/training")
from utils import AddCostToRewardEnv

class GuidePolicyOnly(JumpStartIQL):
    def __init(
        self,
        policy: Union[str, Type[SACPolicy]],
        expert_policy: Any,
        env: Union[GymEnv, str],
        # data_collection_env: GymEnv,
        use_transformer_expert: bool,
        target_return: Optional[float] = None,
        reward_scale: Optional[float] = None,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
    ):
        super().__init__(
            self,
            policy,
            expert_policy,
            env,
            # data_collection_env: GymEnv,
            use_transformer_expert,
            target_return= target_return,
            reward_scale = reward_scale,
            obs_mean = obs_mean,
            obs_std = obs_std
        )
    
    def get_guide_probability(self):
        return 1.
    





def evaluate_model_under_env(training_method, env, policy_load_dir = "",  model_config = {}, start_seed = 10000, episode_len = 90):
    # first update config to test config, including changing agent_policy (in bc), and specify test seed range 
    test_config = {
        "agent_policy":PMKinematicsEgoPolicy,
        "start_seed": start_seed, # seems not working
        "horizon": episode_len/5
    }

    

    # print("start_case_index: before ", env.config["start_case_index"] )
    
    env.config.update(test_config)
    env.config["start_case_index"] = start_seed
    env_test = AddCostToRewardEnv(env.config)
    # print(env.config)
    # print("start_case_index: after ", env.config["start_case_index"] )
    

    # then load policy and evaluate 
    if training_method in (BC, SAC):
        model = training_method("MlpPolicy", env_test)

        model.set_parameters(policy_load_dir)
        header = "-"*10+" Evaluation of " + training_method.__name__ + "-"*10

    elif training_method in (JumpStartIQL, JumpStartSAC, GuidePolicyOnly):
        # should be able to load all useful info from current env.
        keys = ('expert_policy','use_transformer_expert', 'target_return', 'reward_scale', 'obs_mean', 'obs_std')
        assert all_elements_in_dict_keys(keys, model_config), print('Model missing arguments, check keys') 
        model = training_method(
        "MlpPolicy",
        model_config['expert_policy'],
        env_test,
        use_transformer_expert=model_config['use_transformer_expert'],
        target_return=model_config['target_return'],
        reward_scale=model_config['reward_scale'],
        obs_mean=model_config['obs_mean'],
        obs_std=model_config['obs_std'],
        device='cpu'
        )

        header = "-"*10+" Evaluation of " + training_method.__name__ +", use_transformer = ", model_config['use_transformer_expert'],"-"*10


    else:
        print("[eval] method not implememented!")
        return 

    mean_reward, std_reward, mean_success_rate= evaluate_policy(model, env_test, n_eval_episodes=env_test.config['case_num'], deterministic=True, render=False)
    
    print(header)
    print("mean_reward = ", mean_reward)
    print("std_reward = ",std_reward)
    print("mean_success_rate = ", mean_success_rate)

    env.close()


        
        # print("mean_reward, std_reward, mean_success_rate = ", mean_reward, std_reward, mean_success_rate )
        # for seed in range(0, num_scenarios):
        #     plot_waymo_vs_pred(env, model, seed, 'bc', savefig_dir = "examples/metadrive/figs/bc_vs_waymo/diff_action")


def evaluate_guide_policy_only(env, use_transformer_expert, expert_model_dir):
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
    evaluate_model_under_env(GuidePolicyOnly, env, model_config = model_config)
    
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
    
    parser.add_argument('--policy_load_dir', type=str, default = 'examples/metadrive/example_policy/bc-diff-peak.pt')
    
    args = parser.parse_args()
    args = vars(args)
    config =  {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": 100,
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
    
    
    # env.seed(0)

    # test BC 
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(BC, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/bc-diff-peak-10000.pt'
        # )
    # env.close()
    
    # test SAC
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(SAC, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt'
        # )
    # env.close()
    
    # test DT guide policy only:
    env = AddCostToRewardEnv(config)
    evaluate_guide_policy_only(env, use_transformer_expert = True, 
        expert_model_dir = "/home/xinyi/src/decision-transformer/gym/wandb/run-20230822_022210-qlsp8h2g"
        )
    env.close()

    # test BC guide policy(redundant but interesting to try, should have the same result as 'test BC')
    # env = AddCostToRewardEnv(config)
    # evaluate_guide_policy_only(env, use_transformer_expert = False, 
    #     expert_model_dir = 'examples/metadrive/example_policy/bc-diff-peak-10000.pt'
    #     )
    # env.close()

    # test JS-iql, with dt as guide policy 
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(JumpStartIQL, env, policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt')
    # env.close()

    # test JS-iql, with bc as guide policy 
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(JumpStartIQL, env, policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt')
    # env.close()

