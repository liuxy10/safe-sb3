from stable_baselines3 import BC
import gym

import h5py
import os
import numpy as np
from datetime import datetime

from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

from metadrive.engine.asset_loader import AssetLoader
from stable_baselines3 import BC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt


import sys
sys.path.append("/home/xinyi/Documents/UCB/safe-sb3/examples/metadrive")
from utils import AddCostToRewardEnv_base
from utils import get_acc_from_vel



WAYMO_SAMPLING_FREQ = 10



def collect_action_acc_grid_data(args):
    
    '''
    sweep action space [-1,+1]* [-1, +1], with grain 10*10 in default,
    collect corresponding acc data

    '''
    num_dp_per_dim = args['num_dp_per_dim'] 
    num_tested_scenarios = num_dp_per_dim**3
    print("num of scenarios: ", num_tested_scenarios)
    env = AddCostToRewardEnv_base(
    {
        "manual_control": False,
        # "no_traffic": True,
        # "case_num": num_tested_scenarios,
        "random_lane_width": False,
        "map_config": dict(
            type = 'block_num',
            config = 1,
            lane_width= 100
        ),
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": True,
        "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
    }
    )





    env.seed(0)

    '''

    Assume the action is given in the unit of power, this means that the acc generated must depend on a base speed, throttle and steering angle.
    Thus we need to Perform a series of experiments where you vary the speed, throttle position, and steering angle while measuring the corresponding 
    acceleration. Use different combinations of inputs to cover a wide range of driving scenarios.

    The testing procedure: vary the speed, throttle position, and steering angle

    1. drive straight till reaching the base speed, say 5 m/s.
    
    2. collect 2 sec (2*10 = 20 dps) of lontitute and latitute speed data when you maintain the specified throttle position, and steering angle.

    3. assume the base speed does not change during 2 sec, fit a line to the 20 data points collected. the slope would be the estimated acceleration 
    (we could also plot residual vesus time to see if the linearization approximation makes sense)

    4. record the (slope,std) into a num_dp_per_dim*num_dp_per_dim*num_dp_per_dim map 
        
    '''
    
    lat = np.linspace(-1,1,num_dp_per_dim)
    lon = np.linspace(-1,1,num_dp_per_dim)
    speed = [0, 5, 10, ]
    action_lat, action_lon = np.meshgrid(lat,lon)
    lasting_sec = 2

    # collect acc data lasting for 2 sec each 

    acc = np.zeros([num_dp_per_dim, num_dp_per_dim, num_dp_per_dim])
    
    # collect action with specified num of scenarios
    for k in range(0, num_tested_scenarios):
        o = env.reset()
        for i in range(num_dp_per_dim):
            for j in range(num_dp_per_dim):
                vels = []
                headings = []
                local_accs = []
                
                # one round of experiment 
                for _ in range(int(lasting_sec * WAYMO_SAMPLING_FREQ)):
                    
                    action = np.array([action_lat[i,j], action_lon[i,j]])
                    o, r, d, info = env.step(action)
                    vels.append(env.engine.agent_manager.active_agents['default_agent'].velocity)
                    headings.append(env.engine.agent_manager.active_agents['default_agent'].heading_theta)


                # as long as we agree that the heading direction is the same with velocity, 
                # then we can calculate the longtitute and latitute speed as the derivative of the local velocity
                
                headings = np.array(headings)
                vels = np.array(vels)

                local_vel = np.array([vels [:,0]*np.cos(headings) + vels [:,1]*np.sin(headings), 
                                      vels [:,0]*np.sin(headings) - vels [:,1]*np.cos(headings)]).T
                ts = np.arange(int(lasting_sec * WAYMO_SAMPLING_FREQ))/WAYMO_SAMPLING_FREQ
                local_acc = get_acc_from_vel(local_vel,ts, smooth_acc = True) 
                # local_acc = np.array([global_acc [:,0]*np.cos(headings) + global_acc [:,1]*np.sin(headings), 
                #                       global_acc [:,0]*np.sin(headings) - global_acc [:,1]*np.cos(headings)]).T
                
                plt.figure()
                # Plot time versus acceleration
                plt.plot(ts, local_acc[:,0], label='local lat')
                plt.plot(ts, local_acc[:,1], label='local lon')
                # plt.plot(ts, global_acc[:,0], ts, global_acc[:,1], label = 'global')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Acceleration')
                plt.title('Time vs. Acceleration')
                plt.figure()
                # Plot time versus acceleration
                plt.plot(ts, local_vel[:,0], label = 'local lat')
                plt.plot(ts, local_vel[:,1], label = 'local lon')
                # plt.plot(ts, vels[:,0], ts, vels[:,1], label = 'global')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('local Velocity')
                plt.title('Time vs. Velocity')
                plt.show()

                    
                
        print('test '+ str(k) +' is over!')

                        

                
            
    del model
    env.close()





if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)

    parser.add_argument('--num_dp_per_dim', '-num', type=float, default=10)
    args = parser.parse_args()
    args = vars(args)

    collect_action_acc_grid_data(args)
    # test(args)