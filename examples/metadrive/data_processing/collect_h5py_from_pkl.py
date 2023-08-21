
import numpy as np

from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy
# from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading, waymo_2_metadrive_position
from utils import get_acc_from_vel, get_local_from_heading, get_acc_from_speed, get_rate_from_heading

import tqdm
import h5py
import os
import matplotlib.pyplot as plt

import sys
sys.path.append("examples/metadrive/training")
from utils import AddCostToRewardEnv
from utils import estimate_action


WAYMO_SAMPLING_FREQ = 10
TOTAL_TIMESTAMP = 90



def get_current_ego_trajectory_old(waymo_env,i):
    data = waymo_env.engine.data_manager.get_case(i) # 
    
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
    # local_vel = get_local_from_heading(velocity, heading) # not used
    # local_acc = get_acc_from_vel(local_vel, ts)
    acc = get_acc_from_speed(speed, ts)
    heading_speed = get_rate_from_heading(heading, ts)

    # print("max speed, avg speed, acc range,  heading range = {:.{}f}, {:.{}f}, [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}]".format(
    #        np.max(speed), 3, np.mean(speed), 3, np.min(acc), 3, np.max(acc), 3, np.min(heading), 3, np.max(heading), 3))
    

    plot_global_vs_local_vel = False
    if plot_global_vs_local_vel:
        plt.figure()
        # Plot time versus vel
        plt.plot(ts, velocity[:,0], label = 'global x')
        plt.plot(ts, velocity[:,1], label = 'global y')
        # plt.scatter(ts, local_vel[:,0], label = 'local lon')
        # plt.scatter(ts, local_vel[:,1], label = 'local lat')
        # plt.plot(ts, vels[:,0], ts, vels[:,1], label = 'global')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('local/global Velocity')
        plt.title("Time vs. local/global Velocity")
        plt.show()
    
    plot_acc = False
    if plot_acc:
        plt.figure()
        # Plot time versus acc
        plt.scatter(ts, acc, label = 'acc')
        plt.scatter(ts, heading_speed, label = 'heading')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('acc')
        plt.title("Time vs. acc")
        plt.show()

    
    return ts, position, velocity, acc, heading, heading_speed


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
        "use_render": False,
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

    # ## add another four recorded data for action candidates
    headings, heading_rates, speeds, accelerations = np.ndarray((0, 1)),np.ndarray((0, 1)),np.ndarray((0, 1)),np.ndarray((0, 1))

    for seed in range(num_scenarios):
        try: 
            env.reset(force_seed=seed)
            # ts, _, vel, _ = get_current_ego_trajectory(env,seed)
            ts, _, vel, acc, heading, heading_rate = get_current_ego_trajectory_old(env,seed)
            speed = np.linalg.norm(vel, axis = 1)
            # for t in tqdm.trange(acc.shape[0], desc="Timestep"):
            for t in range(acc.shape[0]):
                action = np.array([heading[t], acc[t]]) 

                # whatever the input action is overwrited to be zero (due to the replay policy)
                obs, reward, done, info = env.step(action) 
                obs_rec = np.concatenate((obs_rec, obs.reshape(1, obs.shape[0])))
                ac_rec = np.concatenate((ac_rec, action.reshape(1, action.shape[0])))
                re_rec = np.concatenate((re_rec, np.array([reward])))
                terminal_rec = np.concatenate((terminal_rec, np.array([done])))
                cost_rec = np.concatenate((cost_rec, np.array([info['cost']])))
            
            print("i, max speed, avg speed, acc range,  heading range, reward range, cost range = " +str(seed)+",{:.{}f}, {:.{}f}, [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}]".format(
           np.max(vel), 3, np.mean(vel), 3, np.min(acc), 3, np.max(acc), 3, np.min(heading), 3, np.max(heading), 3, np.min(re_rec), 3, np.max(re_rec), 3, np.min(cost_rec), 3, np.max(cost_rec), 3))
            # add recorded candidates
            headings = np.concatenate((headings, heading.reshape(-1,1)))
            heading_rates = np.concatenate((heading_rates, heading_rate.reshape(-1,1)))
            speeds = np.concatenate((speeds, speed.reshape(-1,1)))
            accelerations = np.concatenate((accelerations, acc.reshape(-1,1)))

        
            num_scenarios_per_buffer = 100
            num_dps_per_scenarios = acc.shape[0]
            num_dps_per_buffer = num_scenarios_per_buffer * num_dps_per_scenarios
            max_num_dps = num_scenarios * num_dps_per_scenarios

            name_dat_dict = {
                                "observation":obs_rec, 
                                "action": ac_rec, 
                                "reward": re_rec, 
                                "terminal":terminal_rec,
                                "cost":cost_rec,
                                "headings":headings, 
                                "heading_rates": heading_rates, 
                                "speeds": speeds, 
                                "accelerations":accelerations
                                # "headings":heading.reshape(-1, 1), 
                                # "heading_rates": heading_rate.reshape(-1, 1), 
                                # "speeds": speed.reshape(-1, 1), 
                                # "accelerations":acc.reshape(-1, 1)
                                }

            if seed % num_scenarios_per_buffer == num_scenarios_per_buffer - 1:
                if seed < num_scenarios_per_buffer:
                    
                    f = h5py.File(args['h5py_path'], 'w') 
                    for name in name_dat_dict.keys():
                        data = name_dat_dict[name]
                        if len(data.shape) == 2:
                            f.create_dataset(name,(num_dps_per_buffer, data.shape[1]), maxshape=(max_num_dps, data.shape[1]), data=data)
                        else: 
                            f.create_dataset(name, (num_dps_per_buffer, ), maxshape=(max_num_dps,), data=data)

                    # Flush the changes to the file
                    f.flush()

                    # reinit all recorded variables
                    obs_rec = np.ndarray((0, ) + env.observation_space.shape)
                    ac_rec = np.ndarray((0, ) + env.action_space.shape)
                    re_rec = np.ndarray((0, ))
                    terminal_rec = np.ndarray((0, ), dtype=bool)
                    cost_rec = np.ndarray((0, ))
                    headings, heading_rates, speeds, accelerations = np.ndarray((0, 1)),np.ndarray((0, 1)),np.ndarray((0, 1)),np.ndarray((0, 1))
                
                else: 
                    f = h5py.File(args['h5py_path'], 'a') 
                    
                    
                    for name in name_dat_dict.keys():
                        new_data = name_dat_dict[name]
                        dataset = f[name]
                        if len(dataset.shape) == 2:
                            dataset.resize((dataset.shape[0] + len(new_data), dataset.shape[1]))  # Resize the dataset to accommodate the new data
                        else: 
                            dataset.resize((dataset.shape[0] + len(new_data),))
                        dataset[-len(new_data):] = new_data  # Append the new data

                        # Flush the dataset and file
                        dataset.flush()
                        f.flush()

                    # reinit all recorded variables
                    obs_rec = np.ndarray((0, ) + env.observation_space.shape)
                    ac_rec = np.ndarray((0, ) + env.action_space.shape)
                    re_rec = np.ndarray((0, ))
                    terminal_rec = np.ndarray((0, ), dtype=bool)
                    cost_rec = np.ndarray((0, ))
                    headings, heading_rates, speeds, accelerations = np.ndarray((0, 1)),np.ndarray((0, 1)),np.ndarray((0, 1)),np.ndarray((0, 1))

    



        except:
            print("skipping traj "+str(seed))
            continue
        
            


    f.close()
    env.close()



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='examples/metadrive/pkl_20')
    parser.add_argument('--h5py_path', type=str, default='examples/metadrive/h5py/pkl20_10.h5py')
    parser.add_argument('--num_of_scenarios', type=str, default='10')
    # parser.add_argument('--map_dir', type = str, default = 'examples/metadrive/map_action_to_acc/log/test.npy')
    args = parser.parse_args()
    args = vars(args)

    # main(args)
    main(args)
    # test(args)


# def test(args):
#     file_list = os.listdir(args['pkl_dir'])
#     if args['num_of_scenarios'] == 'ALL':
#         num_scenarios = len(file_list)
#     else:
#         num_scenarios = int(args['num_of_scenarios'])
#         # num_scenarios = 10 
#     print("num of scenarios: ", num_scenarios) 
#     env = AddCostToRewardEnv(
#     {
#         "manual_control": False,
#         "no_traffic": False,
#         "agent_policy":ReplayEgoCarPolicy,
#         # "agent_policy":EnvInputHeadingAccPolicy,

#         "waymo_data_directory":args['pkl_dir'],
#         "case_num": num_scenarios,
#         "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        
#         "reactive_traffic": False,
#                 # "vehicle_config": dict(
#                 #     show_lidar=True,
#                 #     # no_wheel_friction=True,
#                 #     lidar=dict(num_lasers=0))
#                 "vehicle_config": dict(
#                 # no_wheel_friction=True,
#                 lidar=dict(num_lasers=120, distance=50, num_others=4),
#                 lane_line_detector=dict(num_lasers=12, distance=50),
#                 side_detector=dict(num_lasers=160, distance=50)
#             ),
#     }
#     )
    
#     compare_waymo_tracking = True
    
   
    
    
#     # map = np.load(args['map_dir'])[0]
    

#     for seed in range(num_scenarios):
        
#         # try: 
#         env.reset(force_seed=seed)
#         policy = EnvInputHeadingAccPolicy(obj = env.engine.agent_manager.active_agents['default_agent'],
#                                           seed =seed,
#                                           disable_clip=True)
#         # ts, _, vel, _ = get_current_ego_trajectory(env,seed)
#         ts, pos, vel, acc, heading,_ = get_current_ego_trajectory_old(env,seed)
#         speed = np.linalg.norm(vel, axis = 1)

#         pre_actions = []
#         post_actions = []
#         for t in tqdm.trange(acc.shape[0], desc="Timestep"):
#             pre_action = np.array([heading[t], acc[t]]) #waymo heading and waymo acceleration
#             pre_actions.append(pre_action)
#             env.step(pre_action)
#             ## xinyi: ok we need to verify that it is a good mapping here jul 7th
#             # TODO: verify if the mapped action will follow ego recorded trajectory
        
#             post_action= policy.act('default_agent')
#             post_actions.append(post_action)
#         pre_actions = np.array(pre_actions)
#         post_actions = np.array(post_actions)
        
#         plot_act = True
#         if plot_act:
#             plt.figure()
#             # Plot time versus acc
#             plt.scatter(ts, pre_actions[:,0], label = 'pre steering')
#             plt.scatter(ts, pre_actions[:,1], label = 'pre acceleration')
#             plt.plot(ts, post_actions[:,0], '-',label = 'post steering')
#             plt.plot(ts, post_actions[:,1], '-',label = 'post acceleration')
#             plt.legend()
#             plt.xlabel('Time')
#             plt.ylabel('acc')
#             plt.title("Time vs. acc")
#             plt.show()
        
#         if compare_waymo_tracking:
#             env.close()
#             env_parallel = AddCostToRewardEnv(
#             {
#                 "manual_control": False,
#                 "no_traffic": False,
#                 # "agent_policy":ReplayEgoCarPolicy,
#                 "agent_policy":EnvInputHeadingAccPolicy,

#                 "waymo_data_directory":args['pkl_dir'],
#                 "case_num": num_scenarios,
#                 "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
                
#                 "reactive_traffic": False,
#                         # "vehicle_config": dict(
#                         #     show_lidar=True,
#                         #     # no_wheel_friction=True,
#                         #     lidar=dict(num_lasers=0))
#                         "vehicle_config": dict(
#                         # no_wheel_friction=True,
#                         lidar=dict(num_lasers=120, distance=50, num_others=4),
#                         lane_line_detector=dict(num_lasers=12, distance=50),
#                         side_detector=dict(num_lasers=160, distance=50)
#                     ),
#             }
#             )

#             env_parallel.reset(force_seed=seed)
#             pos_track, speed_track, acc_track, heading_track = [],[],[],[]

#             for t in tqdm.trange(acc.shape[0], desc="Timestep"):
#                 env_parallel.step(post_action)
#                 track_agent = env_parallel.engine.agent_manager.active_agents['default_agent']
#                 pos_track.append(track_agent.position)
#                 speed_track.append(np.linalg.norm(track_agent.velocity))
#                 heading_track.append(track_agent.heading_theta)
            
#             pos_track = np.array(pos_track)

#             fig = plt.figure()
#             ax1 = fig.add_subplot(311)
#             ax2 = fig.add_subplot(312)
#             ax3 = fig.add_subplot(313)
#             # Plot time versus acc
#             ax1.scatter(pos[:,0], pos[:,1], label = 'waymo pos')
#             ax2.scatter(ts, speed, label = 'waymo speed')
#             ax3.scatter(ts, heading, label = 'waymo heading')
#             ax1.scatter(pos_track[:,0], pos_track[:,1], label = 'track pos')
#             ax2.scatter(ts, speed_track, label = 'track speed')
#             ax3.scatter(ts, heading_track, label = 'track heading')
#             ax1.legend()
#             ax2.legend()
#             ax3.legend()
#             ax1.set_xlabel('Time')
#             ax2.set_xlabel('Time')
#             ax3.set_xlabel('Time')
#             ax1.set_ylabel('position x/y')
#             ax2.set_ylabel('speed')
#             ax3.set_ylabel('heading')
#             # ax1.set_title("Time vs. position")
#             plt.show()

#             env_parallel.close()
            
