
import numpy as np

# from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
# from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading, waymo_2_metadrive_position
from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from utils import  get_acc_from_vel
import tqdm
import h5py
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

import tqdm

WAYMO_SAMPLING_FREQ = 10
TOTAL_TIMESTAMP = 90

class AddCostToRewardEnv(WaymoEnv):
    def __init__(self, wrapped_env, lamb=1.):
        """Initialize the class.
        
        Args: 
            wrapped_env: the env to be wrapped
            lamb: new_reward = reward + lamb * cost_hazards
        """
        super().__init__(wrapped_env)
        self._lamb = lamb

    def set_lambda(self, lamb):
        self._lamb = lamb

    def step(self, action):
        state, reward, done, info = super().step(action)
        new_reward = reward - self._lamb * info['cost']
        info["re"] = reward
        return state, new_reward, done, info




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
    print(i, np.max(speed), np.mean(speed))
    
    

    # accoroding to 0.2.6 metadrive waymo_traffic_manager.py, the coodination shift is implemented here:
    position[:,1] = -position[:,1]
    heading = np.rad2deg(-heading)
    velocity[:,1] = -velocity[:,1]
    

    # velocity[:,0] = velocity[:,0]* np.cos(heading)

    # velocity[:,1] = velocity[:,1]* np.sin(heading)



    global_acc = get_acc_from_vel(velocity,ts)
    local_acc = global_acc 
    local_acc[:,0] = global_acc[:,0] * np.cos(-heading)
    local_acc[:,1] = global_acc[:,1] * np.sin(-heading)



    return ts, position, velocity, local_acc, heading








def main(args):

    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        # num_scenarios = int(args['num_of_scenarios'])
        num_scenarios = 10
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
    # cost_hazards_rec = np.ndarray((0, ))
    

    
    f = h5py.File(args['h5py_path'], 'w')
    for seed in range(num_scenarios):
        # try: 
            env.reset(force_seed=seed)
            # ts, _, vel, _ = get_current_ego_trajectory(env,seed)
            ts, _, vel, acc, heading = get_current_ego_trajectory_old(env,seed)
            # 
   
            # try without filtered acc to maintain consistancy with speed

            # acc[:,0] = savgol_filter(acc[:,0], 20, 3)
            # acc[:,1] = savgol_filter(acc[:,1], 20, 3)

            
            plt.figure()
            # # Plot time versus acceleration
            # plt.plot(ts, acc[:,0], ts, acc[:,1])
            # plt.legend()
            # plt.xlabel('Time')
            # plt.ylabel('Acceleration')
            # plt.title('Time vs. Acceleration')
            # plt.figure()
            # # Plot time versus acceleration
            # plt.plot(ts, vel[:,0], ts, vel[:,1])
            # plt.legend()
            # plt.xlabel('Time')
            # plt.ylabel('Velocity')
            # plt.title('Time vs. Velocity')
            # plt.show()


            # PLOT SLIP ANGLE:
            plt.plot(ts, np.arctan2(vel[:,1],vel[:,0])*180 /np.pi, label = 'vel dir')
            plt.plot(ts, heading, label = 'heading')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('heading (deg) and vel_direction (deg)')
            plt.show()
            
            
            for t in tqdm.trange(acc.shape[0], desc="Timestep"):
                # ac = np.array([1.0,1.0]) #dummy action
                ac = acc[t,:] 
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
    parser.add_argument('--num_of_scenarios', type=str, default='3')
    args = parser.parse_args()
    args = vars(args)

    # main(args)
    main(args)
