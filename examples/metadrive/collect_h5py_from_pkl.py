
import numpy as np

# from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
# from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading, waymo_2_metadrive_position
from utils import get_acc_from_vel, get_local_from_heading

import tqdm
import h5py
import os
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/xinyi/Documents/UCB/safe-sb3/examples/metadrive")
from utils import AddCostToRewardEnv
from utils import estimate_action
sys.path.append("/home/xinyi/Documents/UCB/safe-sb3/examples/metadrive/map_action_to_acc")
from visualize_map import plot_reachable_region

WAYMO_SAMPLING_FREQ = 10
TOTAL_TIMESTAMP = 90

def get_current_ego_trajectory(waymo_env):
    data = waymo_env.engine.data_manager.current_scenario

    id = data['metadata']['sdc_id']
    position = np.array(data['tracks'][id]['state']['position'])
    heading = np.array(data['tracks'][id]['state']['heading'])
    velocity = np.array(data['tracks'][id]['state']['velocity'])
    ts = np.array(data['metadata']['ts'])
    return ts, position, velocity, heading

def get_current_ego_trajectory_old(waymo_env,i):
    data = waymo_env.engine.data_manager.get_case(i)
    
    sdc_id = data['sdc_index']
    state_all_traj = data['tracks'][sdc_id]['state'] # 199*10

    ts = np.array(data['ts'])
    position = np.array(state_all_traj[:,0:2])
    heading = np.array(state_all_traj[:,6])
    velocity = np.array(state_all_traj[:,7:9])
    speed = np.linalg.norm(velocity, axis = 1)
    
    
    

    # accoroding to 0.2.6 metadrive waymo_traffic_manager.py, the coodination shift is implemented here:
    position[:,1] = -position[:,1]
    heading = -heading
    velocity[:,1] = -velocity[:,1]
    
    # revised to be consistant with collect_action_acc_pair.py
    local_vel = get_local_from_heading(velocity, heading)
    local_acc = get_acc_from_vel(local_vel, ts)

    print("max speed, avg speed, max lat acc, max lon acc, min lon acc = {:.{}f}, {:.{}f}, {:.{}f}, {:.{}f}, {:.{}f}".format(
           np.max(speed), 3, np.mean(speed), 3, np.max(np.abs(local_acc[:,1])), 3, np.max(local_acc[:,0]), 3, np.min(local_acc[:,0]), 3))
    

    plot_global_vs_local_vel = False
    if plot_global_vs_local_vel:
        plt.figure()
        # Plot time versus vel
        plt.plot(ts, velocity[:,0], label = 'global x')
        plt.plot(ts, velocity[:,1], label = 'global y')
        plt.scatter(ts, local_vel[:,0], label = 'local lon')
        plt.scatter(ts, local_vel[:,1], label = 'local lat')
        # plt.plot(ts, vels[:,0], ts, vels[:,1], label = 'global')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('local/global Velocity')
        plt.title("Time vs. local/global Velocity")
        plt.show()

    return ts, position, velocity, local_acc, heading


def main(args):



    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        num_scenarios = int(args['num_of_scenarios'])
        # num_scenarios = 10 
    print("num of scenarios: ", num_scenarios) 
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":ReplayEgoCarPolicy,
        "waymo_data_directory":args['pkl_dir'],
        "case_num": num_scenarios,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        
        "reactive_traffic": False,
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
    }
    )

    # init all recorded variables
    obs_rec = np.ndarray((0, ) + env.observation_space.shape)
    ac_rec = np.ndarray((0, ) + env.action_space.shape)
    re_rec = np.ndarray((0, ))
    terminal_rec = np.ndarray((0, ), dtype=bool)
    cost_rec = np.ndarray((0, ))

    
    f = h5py.File(args['h5py_path'], 'w')
    map = np.load(args['map_dir'])[0]

    for seed in range(num_scenarios):
        # try: 
            env.reset(force_seed=seed)
            # ts, _, vel, _ = get_current_ego_trajectory(env,seed)
            ts, _, vel, acc, heading = get_current_ego_trajectory_old(env,seed)
            
            

            speed = np.linalg.norm(vel, axis = 1)


            plot_traj_range = False
            if plot_traj_range:
                plot_reachable_region(speed, acc[:,1], acc[:,0])
            
           
            plot_slip_angle_gap =False
            if plot_slip_angle_gap:
                plt.figure()
                plt.plot(ts, np.arctan2(vel[:,1],vel[:,0]), label = 'vel dir')
                plt.plot(ts, heading, label = 'heading')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('heading (rad) and vel_direction (rad)')
                plt.show()
            
            plot_acc_vel = False
            if plot_acc_vel:
                plt.figure()
                plt.plot(ts, acc[:,0], ts, acc[:,1])
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Acceleration')
                plt.title('Time vs. Acceleration')
                
                plt.figure()
                plt.plot(ts, vel[:,0], ts, vel[:,1])
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Velocity')
                plt.title('Time vs. Velocity')
                plt.show()

            
            for t in tqdm.trange(acc.shape[0], desc="Timestep"):
                # ac = np.array([1.0,1.0]) #dummy action
                lon_acc, lat_acc = acc[t,:]
                
                ac = estimate_action(map, speed[t], lat_acc, lon_acc)
                obs, reward, done, info = env.step(ac) # whatever the input action is overwrited to be zero
                obs_rec = np.concatenate((obs_rec, obs.reshape(1, obs.shape[0])))
                ac_rec = np.concatenate((ac_rec, ac.reshape(1, ac.shape[0]))) 
                re_rec = np.concatenate((re_rec, np.array([reward])))
                terminal_rec = np.concatenate((terminal_rec, np.array([done])))
                cost_rec = np.concatenate((cost_rec, np.array([info['cost']])))
                # env.render(mode="topdown")
                # print(env.vehicle.speed, env.vehicle.heading, reward, info['cost'])
        # except:
        #     continue
        

    f.create_dataset("observation", data=obs_rec)
    f.create_dataset("action", data=ac_rec)
    f.create_dataset("reward", data=re_rec)
    f.create_dataset("terminal", data=terminal_rec)
    f.create_dataset("cost", data=cost_rec)
    env.close()



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', '-e', type=str, default='Safexp-CarButton1-v0')
    # parser.add_argument('--env_seed', '-es', type=int, default=3)
    # parser.add_argument('--steps', type=int, default=int(1e5))
    # parser.add_argument('--policy_load_dir', type=str)
    parser.add_argument('--pkl_dir', type=str, default='examples/metadrive/pkl_9')
    parser.add_argument('--h5py_path', type=str, default='examples/metadrive/h5py/one_pack_from_tfrecord.h5py')
    parser.add_argument('--num_of_scenarios', type=str, default='10')
    parser.add_argument('--map_dir', type = str, default = 'examples/metadrive/map_action_to_acc/log/test.npy')
    args = parser.parse_args()
    args = vars(args)

    # main(args)
    main(args)
