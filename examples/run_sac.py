from stable_baselines3 import SAC
import gym
import safety_gym
import bullet_safety_gym
from wrappers import AddCostToRewardEnv
import h5py
import numpy as np
import os
from datetime import datetime


def main(args):
    env_name = args["env"]
    env = gym.make(env_name)
    env.seed(args["env_seed"])
    if args["random"]:
        env.set_num_different_layouts(100)
    lamb = args["lambda"]
    env = AddCostToRewardEnv(env, lamb=lamb)

    root_dir = "tensorboard_logs"
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = (
        "sac-" + env_name + "_es" + str(args["env_seed"]) 
        # + "_lam" + str(lamb) + '_' + date)
        + "_lam" + str(lamb))
    if args["suffix"]:
        experiment_name += f'_{args["suffix"]}'
    tensorboard_log = os.path.join(root_dir, experiment_name)

    model = SAC("MlpPolicy", env, tensorboard_log=tensorboard_log, verbose=1, device="cpu")
    model.learn(total_timesteps=args["steps"])

    del model

    env.close()


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-CarButton1-v0')
    parser.add_argument('--env_seed', '-es', type=int, default=3)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--steps', '-st', type=int, default=int(1e7))
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    args = vars(args)

    main(args)