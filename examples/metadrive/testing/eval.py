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


class GuidePolicyOnly(JumpStartIQL):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        expert_policy: Any,
        env: Union[GymEnv, str],
        use_transformer_expert: bool,
        target_return: Optional[float] = None,
        reward_scale: Optional[float] = None,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        guidance_timesteps: int = 500_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            expert_policy,
            env,
            use_transformer_expert,
            target_return = target_return,
            reward_scale = reward_scale,
            obs_mean = obs_mean,
            obs_std = obs_std,
            guidance_timesteps = guidance_timesteps,
            learning_rate = learning_rate,
            buffer_size = buffer_size,  # 1e6
            learning_starts = learning_starts,
            batch_size = batch_size,
            tau = tau,
            gamma = gamma,
            train_freq = train_freq,
            gradient_steps = gradient_steps,
            action_noise = action_noise,
            replay_buffer_class = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            optimize_memory_usage = optimize_memory_usage,
            ent_coef = ent_coef,
            target_update_interval = target_update_interval,
            target_entropy = target_entropy,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            use_sde_at_warmup = use_sde_at_warmup,
            stats_window_size = stats_window_size,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            seed = seed,
            device = device,
            _init_setup_model= _init_setup_model
        )

        if self._last_obs is None:
            assert self.env is not None
            # pytype: disable=annotation-type-mismatch
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            # Historical info within one episode
            assert len(self.observation_space.shape) == 1
            assert len(self.action_space.shape) == 1
            self.obs_dim = self.observation_space.shape[0]
            self.ac_dim = self.action_space.shape[0]
            self.hist_obs = self._last_obs.reshape(1, self.obs_dim)
            self.hist_ac= np.zeros((0, self.ac_dim))
            self.hist_re = np.zeros(0)
            if (
                hasattr(self, 'use_transformer_expert') 
                and self.use_transformer_expert
            ):
                self.target_return = np.array([[self.target_return_init]])
            self.timesteps = np.zeros((1, 1))

    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.hist_ac  = np.concatenate(
            [self.hist_ac, np.zeros((1, self.ac_dim))], axis=0
        )
        self.hist_re = np.concatenate([self.hist_re, np.zeros(1)])
        if self.use_transformer_expert:
            hist_obs = th.tensor(
                self.hist_obs, dtype=th.float32, device=self.device
            )
            hist_ac = th.tensor(
                self.hist_ac, dtype=th.float32, device=self.device
            )
            hist_re = th.tensor(
                self.hist_re, dtype=th.float32, device=self.device
            )
            target_return = th.tensor(
                self.target_return, dtype=th.float32, device=self.device
            )
            timesteps = th.tensor(
                self.timesteps, dtype=th.long, device=self.device
            )
            unscaled_action = self.expert_policy.get_action(
                (hist_obs - self.obs_mean) / self.obs_std,
                hist_ac,
                hist_re,
                target_return,
                timesteps,
            )
            unscaled_action = unscaled_action.detach().cpu().numpy()
            unscaled_action = unscaled_action.reshape(1,2)
            # print("[use_guide] unscaled_action.shape", unscaled_action.shape)
        else:
            unscaled_action, _ = self.expert_policy.predict(
                self._last_obs, deterministic=False
            )
        

        return unscaled_action, None






        return 

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
        

    elif training_method in (JumpStartIQL, JumpStartSAC, GuidePolicyOnly):
        # should be able to load all useful info from current env.
        keys = ('expert_policy','use_transformer_expert', 'target_return', 'reward_scale', 'obs_mean', 'obs_std')
        assert all_elements_in_dict_keys(keys, model_config), print('Model missing arguments, check keys') 
        model = training_method(
        "MlpPolicy",
        model_config['expert_policy'],
        env_test,
        use_transformer_expert = model_config['use_transformer_expert'],
        target_return=model_config['target_return'],
        reward_scale=model_config['reward_scale'],
        obs_mean=model_config['obs_mean'],
        obs_std=model_config['obs_std'],
        device='cpu'
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
    
    parser.add_argument('--policy_load_dir', type=str, default = 'examples/metadrive/example_policy/bc-diff-peak.pt')
    
    args = parser.parse_args()
    args = vars(args)
    config =  {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": 10,
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
    
    save_fig_dir = "/home/xinyi/src/safe-sb3/examples/metadrive/figs/"
    
    # test BC 
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(BC, env, 
    #     policy_load_dir = 'examples/metadrive/example_policy/bc-diff-peak-10000.pt',
    #     save_fig_dir = save_fig_dir
    #     )
    # env.close()
    
    # test SAC
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(SAC, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt',
        # save_fig_dir = save_fig_dir
        # )
    # env.close()
    
    # test DT guide policy only:
    env = AddCostToRewardEnv(config)
    evaluate_guide_policy_only(env, use_transformer_expert = True, 
        expert_model_dir = "/home/xinyi/src/decision-transformer/gym/wandb/run-20230822_180622-20swd1g8",
        save_fig_dir = save_fig_dir
        )
    env.close()

    # test BC guide policy(redundant but interesting to try, should have the same result as 'test BC')
    env = AddCostToRewardEnv(config)
    evaluate_guide_policy_only(env, use_transformer_expert = False, 
        expert_model_dir = 'examples/metadrive/example_policy/bc-diff-peak-10000.pt',
        save_fig_dir = save_fig_dir
        )
    env.close()

    # test JS-iql, with dt as guide policy 
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(JumpStartIQL, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt',
        # save_fig_dir = save_fig_dir
        # )
    # env.close()

    # test JS-iql, with bc as guide policy 
    # env = AddCostToRewardEnv(config)
    # evaluate_model_under_env(JumpStartIQL, env, 
        # policy_load_dir = 'examples/metadrive/example_policy/sac-diff-peak-1000.pt',
        # save_fig_dir = save_fig_dir
        # )
    # env.close()

