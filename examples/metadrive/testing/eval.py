from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import JumpStartIQL, BC, JumpStartSAC, SAC

from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy

import sys

sys.path.append("examples/metadrive/training")
from utils import AddCostToRewardEnv



def evaluate_model_under_env(training_method, env, policy_load_dir,  training_config = {}, start_seed = 10000, episode_len = 90):
# first update config to test config, including changing agent_policy (in bc), and specify test seed range
        test_config = {
            "agent_policy":PMKinematicsEgoPolicy,
            "start_seed": start_seed,
            "horizon": episode_len/5
        }
        env.config.update(test_config)
        # print(env.config)

        # then load policy and evaluate 
        if training_method in (BC, SAC):
            model = training_method("MlpPolicy", env)

            model.set_parameters(policy_load_dir)
            print("-"*10+" Evaluation of " + training_method.__name__ + "-"*10)

        elif training_method in (JumpStartIQL, JumpStartSAC):
            # should be able to load all useful info from current env.
            keys = ('expert_policy','use_transformer_expert', 'target_return', 'reward_scale', 'obs_mean', 'obs_std', 'device')
            assert all_elements_in_dict_keys(keys, config), print('Model missing arguments, check keys') 
            model = training_method(
            "MlpPolicy",
            training_config['expert_policy'],
            env,
            use_transformer_expert=training_config['use_transformer_expert'],
            target_return=training_config['target_return'],
            reward_scale=training_config['reward_scale'],
            obs_mean=training_config['obs_mean'],
            obs_std=training_config['obs_std'],
            device=training_config['device'])

            print("-"*10+" Evaluation of " + training_method.__name__ +", use_transformer = ", use_transformer_expert,"-"*10)
    

        else:
            print("[eval] method not implememented!")
            return 

        mean_reward, std_reward, mean_success_rate=evaluate_policy(model, env, n_eval_episodes=50, deterministic=True, render=False)
        
        print("mean_reward = ", mean_reward)
        print("std_reward = ",std_reward)
        print("mean_success_rate = ", mean_success_rate)
        
        # print("mean_reward, std_reward, mean_success_rate = ", mean_reward, std_reward, mean_success_rate )
        # for seed in range(0, num_scenarios):
        #     plot_waymo_vs_pred(env, model, seed, 'bc', savefig_dir = "examples/metadrive/figs/bc_vs_waymo/diff_action")






def all_elements_in_dict_keys(elements, dictionary):
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
    # print(os.listdir(args['pkl_dir']))
    env = AddCostToRewardEnv(
    {
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
    },
    )
    env.seed(0)

    # test BC 
    # evaluate_model_under_env(BC, env, policy_load_dir = 'examples/metadrive/example_policy/bc-diff-peak-10000.pt')
    
    
    # test SAC
    evaluate_model_under_env(SAC, env, policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt')
    
