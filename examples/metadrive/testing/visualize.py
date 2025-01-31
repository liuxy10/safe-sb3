
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy

from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
from stable_baselines3 import BC, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
# what about the data from h5py
import h5py
import sys
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/data_processing")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
from utils import AddCostToRewardEnv
from combine_pkls_for_dt import collect_rollout_in_one_seed

# sys.path.append("/home/xinyi/src/decision-transformer/gym/decision_transformer/evaluation")

from plot_utils import plot_states_compare

# import asciiplotlib as apl

# specification for 

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

    

    plot_rew_cost_versus_time = False
    if plot_rew_cost_versus_time:
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

    plot_hist_candidate_actions = True
    if plot_hist_candidate_actions:
        dataset_names = ['headings', 'heading_rates','speeds','accelerations']
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i, ax in enumerate(axes.flatten()):
            data_name = dataset_names[i]
            data = np.array(hf[data_name])
            ax.hist(data, bins=50, density = True, log=True)
            ax.set_title(data_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            plt.show()

def plot_waymo_vs_pred(env, model,seed, md_name, savefig_dir="", save_render_dir="", end_eps_when_done = True):
    from collect_h5py_from_pkl import get_current_ego_trajectory_old
    
    o = env.reset(force_seed=seed)
    
    #recorded ts, position, velocity, acc, heading from waymo
    ts, pos_rec, vel_rec, acc_rec, heading_rec, heading_rate_rec = get_current_ego_trajectory_old(env,seed)
    speed_rec = np.linalg.norm(vel_rec, axis = 1)
    # print(np.array(env.engine.agent_manager.active_agents['default_agent'].speed), speed_rec[0])

    
    actual_pos = np.zeros_like(pos_rec)
    action_pred = np.zeros_like(pos_rec) # default diff action version
    actual_heading = np.zeros_like(ts)
    actual_speed= np.zeros_like(ts)
    actual_rew = np.zeros_like(ts)


    cum_rew, cum_cost = 0,0
    for i in range(len(ts)):
        if md_name in ['cvpo', 'js-cvpo']:
            res = model.act(o, deterministic=True, with_logprob=False)
            action = res[0]
            o, r, done, info = env.step(action)
        else:
            action, _ = model.predict(o, deterministic = True)
            # action = [0, 4]
            o, r, done, info = env.step(action)
        actual_heading[i] = env.engine.agent_manager.active_agents['default_agent'].heading_theta
        actual_speed[i] = np.array(env.engine.agent_manager.active_agents['default_agent'].speed/3.6)
        actual_pos [i,:] = np.array(env.engine.agent_manager.active_agents['default_agent'].position)
        action_pred [i,:] = action
        actual_rew[i] = r
        cum_rew += r
        # save
        rgb_cam = env.vehicle.image_sensors[env.vehicle.config["image_source"]]
        rgb_cam.save_image(env.vehicle, name="{}.png".format(i))
        
        cum_cost += info['cost']
        # print('seed:', seed, 'step:', i,'action:', action, 'reward: ', r, 'cost: ',info['cost'],'cum reward: ', cum_rew, 'cum cost: ',cum_cost, 'done:', d)
        if end_eps_when_done and done:
            actual_heading[i+1:] = None
            actual_speed[i+1:] = None
            action_pred[i+1:] = None
            actual_rew[i+1:] = None

            actual_pos[i+1:,:] = None

            break

    # print(f"avg action error (heading rate) = {np.mean((action_pred[0] - heading_rate_rec)**2)}")
    # print(f"avg action error (accel)        = {np.mean((action_pred[1] - acc_rec)**2)}")
    
    plot_comparison = True
    action_pred = np.array(action_pred)
    if plot_comparison:
        plot_states_compare(ts, 
                   action_pred, acc_rec, 
                   actual_speed, speed_rec, 
                   pos_rec, actual_pos, 
                   actual_heading, heading_rec, 
                   actual_rew,
                   save_fig_dir = savefig_dir,
                   seed=seed, 
                   succeed = info['arrive_dest'],
                   md_name = md_name)
        # pos_pred = np.array(pos_pred)
        # fig, axs = plt.subplots(2, 2)

        # axs[0,0].plot(ts, action_pred[:,1], label = md_name+' pred acc')
        # axs[0,0].plot(ts, acc_rec, label = 'waymo acc' )
        # axs[1,0].plot(ts, actual_heading, label = md_name+' actual heading' )
        # axs[1,0].plot(ts, heading_rec, label = 'waymo heading')
        # axs[0,1].plot(ts, actual_speed, label = md_name+' actual speed' )
        # axs[0,1].plot(ts, speed_rec, label = 'waymo speed')
        # axs[1,1].plot(ts, actual_rew, label = md_name+' actual reward' )
        # # axs[1,1].plot(ts, rew_rec, label = 'waymo reward')
        # for i in range(2):
        #     for j in range(2):
        #         axs[i,j].legend()
        #         axs[i,j].set_xlabel('time')
        # axs[0,0].set_ylabel('acceleration')
        # axs[1,0].set_ylabel('heading')
        # axs[0,1].set_ylabel('speed')
        # axs[1,1].set_ylabel('reward')
        # # plt.title("recorded action vs test predicted action")
        # if len(savefig_dir) > 0:
        #     if not os.path.isdir(savefig_dir):
        #         os.makedirs(savefig_dir)
        #     plt.savefig(os.path.join(savefig_dir, "seed_"+str(seed)+".jpg"))
        # else:
        #     plt.show()
    return cum_rew, cum_cost


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
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": True,
            "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    },
    )


    env.seed(args["env_seed"])
    
    
    model = BC.load(args['model_path'])
    # model = SAC.load(args['model_path'])
    for seed in range(0, num_scenarios):
            o = env.reset(force_seed=seed)
            #ts, position, velocity, acc, heading
            ts, pos_rec,_, acc, heading,_ = get_current_ego_trajectory_old(env,seed)
            

            
            
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
                fig, axs = plt.subplot(2,1)
                # Plot pos
                # axs[0,0].plot(pos_pred[:,0], pos_pred[:,1], label = 'test')
                # axs[0,0].plot(pos_rec[:,0], pos_rec[:,1], label = 'record')
                axs[0,0].plot(ts, acc, label = 'waymo acc')
                axs[0,0].plot(ts, action_pred[:,1], label = 'bc pred acc')
                axs[1,0].plot(ts, heading, label = 'waymo heading')
                axs[1,0].plot(ts, action_pred[:,0], label = 'bc pred heading')
                axs[1,0].plot(ts, actual_heading, label = 'actual heading' )
                axs[0,0].legend()
                axs[1,0].legend()
                # axs[0,0].axis('equal')
                axs[0,0].set_xlabel('time')
                axs[0,0].set_ylabel('acceleration')
                axs[1,0].set_xlabel('time')
                axs[1,0].set_ylabel('heading')
                plt.title("recorded action vs test predicted action")
                plt.show()
         
            
    del model
    env.close()

def show_negative_reward_scenarios(args, ill_seeds):
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
        "agent_policy":ReplayEgoCarPolicy, # BC uses ReplayEgoCarPolicy to train policy
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": True,
        "reactive_traffic": False,
               "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    }, 
    )


    env.seed(args["env_seed"])
    
    # model = SAC.load(args['model_path'])
    for seed in ill_seeds[:,0]:
            seed = int(seed)
            if seed < num_scenarios:
                data = collect_rollout_in_one_seed(env, seed)
    env.close()











if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5py_path', type=str, default='examples/metadrive/h5py/bc_9_900.h5py')
    parser.add_argument('--pkl_dir', '-pkl', type=str, default='examples/metadrive/pkl_9_10000')
    parser.add_argument('--model_path', '-out', type=str, default='examples/metadrive/example_policy/bc-waymo-es0.zip')
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=str, default="900")

    args = parser.parse_args()
    args = vars(args)
    # visualize_bc_prediction(args)
    # visualize_h5py(args)
    
    # ill_seeds = np.load("ill_seed.npy")
    # show_negative_reward_scenarios(args, ill_seeds)



    test_env = AddCostToRewardEnv(
        {
            "manual_control": False,
            "no_traffic": False,
            "agent_policy":PMKinematicsEgoPolicy,
            "waymo_data_directory":'/home/xinyi/src/data/metadrive/pkl_9',
            "case_num": 100,
            "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
            "use_render": False,
            'start_seed': 10000,
            "horizon": 90/5,
            "reactive_traffic": False,
                    "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
                lane_line_detector=dict(num_lasers=12, distance=50), # 12
                side_detector=dict(num_lasers=20, distance=50)) # 160,
        },    
    )
    for seed in [10040, 10049]:
        plot_waymo_vs_pred(test_env, None, seed, 'increase speed', '/home/xinyi/src/safe-sb3/examples/metadrive/figs/test')