import json
import os
import re

import numpy as np
import torch
import yaml

from stable_baselines3.bc import BC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.decision_transformers import DecisionTransformer
from stable_baselines3.sac.sac import SAC


def infer_algo_cls_from_dir_name(dir):
    tmp = dir.lower()
    tmp = re.split('_|-|/', tmp)
    if "bc" in tmp:
        return BC
    elif "sac" in tmp:
        return SAC
    else:
        raise NotImplementedError(
            "The expert policy class is not found! Please follow the"
            " convention to name the model directory!"
        )

def load_expert_policy(model_dir, env, device="cpu"):
    algo_cls = infer_algo_cls_from_dir_name(model_dir)
    model = algo_cls("MlpPolicy", env, device=device)
    # model_dir = os.path.join("tensorboard_logs", model_dir, "model.pt")
    model.set_parameters(model_dir)
    expert_policy = model.policy
    return expert_policy

def check_expert_policy(expert_policy, algorithm):
    if isinstance(expert_policy, DecisionTransformer):
        return
    if not hasattr(expert_policy, "predict"):
        raise NotImplementedError(
            "Expert policy must implement a method 'perdict'!"
            " It predicts next actions from current observations."
            " Please follow the definition of `BaseAlgorithm.predict`."
        )
    if not isinstance(algorithm, BaseAlgorithm):
        raise ValueError(
            "`algorithm` must be inherited from `BaseAlgorithm!`"
        )
    expected_shape = (algorithm.n_envs, ) + algorithm.action_space.shape
    obs_shape = (algorithm.n_envs, ) + algorithm.observation_space.shape
    dummy_obs = np.random.rand(*obs_shape)
    dummy_action, _ = expert_policy.predict(dummy_obs)
    if dummy_action.shape != expected_shape:
        raise ValueError(
            "The output shape of `expert_policy.predict` should be"
            f" {expected_shape}! Got {dummy_action.shape} instead."
        )
    
def load_demo_stats(path):
    stats_path = os.path.join(path, 'obs_stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    ## TODO: delete this try when new version of model is loaded:
    try:
        return (
            np.array(stats['obs_mean']), 
            np.array(stats['obs_std']),
            stats['reward_scale'],
            stats['target_return'],
        )
    except:
        return (
            np.array(stats['obs_mean']), 
            np.array(stats['obs_std']),
            None,
            None,
        )

def load_transformer(model_dir, device):
    if model_dir.split('/')[-1] == 'model.pt':
        raise ValueError("Please use the root dir of model.pt!")
    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    # # TODO: delete this part after adding these three into variant and save them in dt
    # if config['env']['value'] == 'waymo':
    #     state_dim= 145
    #     act_dim = 2
    #     max_ep_len = 90
    # else:
    state_dim = config['state_dim']['value']
    act_dim = config['act_dim']['value']
    max_ep_len = config['max_ep_len']['value']
    # print("[js_sac] max_ep_len, config['K']['value']", max_ep_len, config['K']['value'])
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=config['K']['value'],
        max_ep_len=max_ep_len,
        hidden_size=config['embed_dim']['value'], # default 128
        n_layer=config['n_layer']['value'],
        n_head=config['n_head']['value'],
        n_inner=4*config['embed_dim']['value'],
        activation_function=config['activation_function']['value'],
        n_positions=1024,
        resid_pdrop=config['dropout']['value'],
        attn_pdrop=config['dropout']['value'],
    )
    state_dict_path = os.path.join(model_dir, 'model.pt')
    # import pdb; pdb.set_trace()
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()
    return model