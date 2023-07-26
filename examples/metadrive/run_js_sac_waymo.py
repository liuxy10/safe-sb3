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

from stable_baselines3 import JumpStartSAC
from stable_baselines3.js_sac import utils as js_utils


from utils import AddCostToRewardEnv
from visualize import plot_waymo_vs_pred
import matplotlib.pyplot as plt
WAYMO_SAMPLING_FREQ = 10
def main(args):
    device = args["device"]

    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        num_scenarios = int(args['num_of_scenarios'])
    lamb = args["lambda"]
    print("num of scenarios: ", num_scenarios)
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": False,
                "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
    }, lamb= lamb
    )
    env.seed(args["env_seed"])
    if args["random"]:
        env.set_num_different_layouts(100)


    root_dir = "tensorboard_logs"
    experiment_name = (
        "js-sac-waymo_es" + str(args["env_seed"])
        + "_lamb" + str(lamb))
    if args["suffix"]:
        experiment_name += f'_{args["suffix"]}'
    tensorboard_log = os.path.join(root_dir, experiment_name)

    expert_policy = js_utils.load_expert_policy(
        model_dir=args['expert_model_dir'], env=env, device=device
    )

    model = JumpStartSAC(
        "MlpPolicy",
        expert_policy,
        env, env,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
    )
    model.learn(total_timesteps=args["steps"])

    del model
    env.close()

    # done = False
    # while not done:
    #     env.render()
    #     action = env.action_space.sample()  # Replace with your agent's action selection logic
    #     obs, reward, done, info = env.step(action)


    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_20')
    parser.add_argument('--output_dir', '-out', type=str, default='examples/metadrive/saved_sac_policy')
    parser.add_argument('--use_diff_action_space', '-diff', type=bool, default=True)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--device', '-d', type=str, default="cpu")
    parser.add_argument('--expert_model_dir', '-emd', type=str, default='tensorboard_log/bc-waymo-es0/BC_57/model.pt')

    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=str, default="10")
    parser.add_argument('--steps', '-st', type=int, default=int(1e7))
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    args = vars(args)

    main(args)