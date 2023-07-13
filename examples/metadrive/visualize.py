from jupyterlab_h5web import H5Web
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy
from stable_baselines3 import BC
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from utils import AddCostToRewardEnv
import matplotlib.pyplot as plt
# what about the data from h5py
import h5py

WAYMO_SAMPLING_FREQ = 10

def visualize_h5py(args):
    # Specify the filename of the h5py file
    h5py_filename = args['h5py_path']

    hf = h5py.File(h5py_filename, 'r')
    # Get a list of dataset names
    dataset_names = list(hf.keys())

    print("Names of h5py datasets:")
    for name in dataset_names:
        print(name)

    re = np.array(hf["reward"])
    cost = np.array(hf["cost"])
    ac = np.array(hf["action"])
    plt.figure()
    # Plot time versus acceleration
    plt.plot(range(len(re)), re, label = "reward")
    plt.plot(range(len(cost)), cost, label = "cost")
    plt.legend()
    plt.xlabel('TimeStamp')
    plt.ylabel('Reward/cost')
    plt.title('TimeStamp vs. Reward/cost')
    plt.figure()
    # Plot time versus acceleration
    plt.plot(np.arange(ac.shape[0]), ac[:,0], label = 'heading')
    plt.plot(np.arange(ac.shape[0]), ac[:,1], label = 'acceleration')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('action value')
    plt.title('TimeStamp vs. action')
    plt.show()


def visualize_bc_prediction(args):
    from collect_h5py_from_pkl import get_current_ego_trajectory_old

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
        "agent_policy":EnvInputHeadingAccPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": True,
        "reactive_traffic": True,
                # "vehicle_config": dict(
                #     show_lidar=True,
                #     # no_wheel_friction=True,
                #     lidar=dict(num_lasers=0))
                "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
    },
    )


    env.seed(args["env_seed"])
    
    exp_name = "bc-waymo-es" + str(args["env_seed"])
    
    model = BC.load(args['model_path'])
    for seed in range(0, num_scenarios):
            o = env.reset(force_seed=seed)
            #ts, position, velocity, acc, heading
            ts, pos_rec,_, acc, heading = get_current_ego_trajectory_old(env,seed)
            
            
            pos_pred = np.zeros_like(pos_rec)
            action_pred = np.zeros_like(pos_rec)
            actual_heading = np.zeros_like(ts)

            cum_rew, cum_cost = 0,0
            for i in range(len(ts)):
                action, _ = model.predict(o, deterministic = True)
                o, r, d, info = env.step(action)
                actual_heading[i] = env.engine.agent_manager.active_agents['default_agent'].heading_theta
                pos_cur = np.array(env.engine.agent_manager.active_agents['default_agent'].position)
                vel_cur = np.array(env.engine.agent_manager.active_agents['default_agent'].velocity)
                pos_pred [i,:] = pos_cur
                action_pred [i,:] = action
                cum_rew += r
                cum_cost += info['cost']
                print('seed:', seed, 'step:', i,'action:', action, 'reward: ', r, 'cost: ',info['cost'],'cum reward: ', cum_rew, 'cum cost: ',cum_cost, 'done:', d)
                
            plot_comparison = True
            action_pred = np.array(action_pred)
            if plot_comparison:
                pos_pred = np.array(pos_pred)
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                # Plot pos
                # ax1.plot(pos_pred[:,0], pos_pred[:,1], label = 'test')
                # ax1.plot(pos_rec[:,0], pos_rec[:,1], label = 'record')
                ax1.plot(ts, acc, label = 'waymo acc')
                ax1.plot(ts, action_pred[:,1], label = 'bc pred acc')
                ax2.plot(ts, heading, label = 'waymo heading')
                ax2.plot(ts, action_pred[:,0], label = 'bc pred heading')
                ax2.plot(ts, actual_heading, label = 'actual heading' )
                ax1.legend()
                ax2.legend()
                # ax1.axis('equal')
                ax1.set_xlabel('time')
                ax1.set_ylabel('acceleration')
                ax2.set_xlabel('time')
                ax2.set_ylabel('heading')
                plt.title("recorded action vs test predicted action")
                plt.show()

            

                






                
            
    del model
    env.close()










if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5py_path', type=str, default='examples/metadrive/h5py/pkl9_900.h5py')
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_9')
    parser.add_argument('--model_path', '-out', type=str, default='examples/metadrive/saved_bc_policy/bc-waymo-es0_500000_steps.zip')
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=str, default="100")

    args = parser.parse_args()
    args = vars(args)
    visualize_bc_prediction(args)