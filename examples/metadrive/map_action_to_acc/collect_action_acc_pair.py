from stable_baselines3 import BC
import gym

import h5py
import os
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt



import sys
sys.path.append("examples/metadrive/training")
from utils import AddCostToRewardEnv_base
from utils import get_acc_from_vel, get_local_from_heading, estimate_action
np.set_printoptions(precision=3)

SAMPLING_FREQ = 10 # we are using sampling frequency of 10 hrz.


def pid_controller(setpoint, measured_value, error_sum, last_error, dt):
     
    # PID Controller Parameters
    Kp = 0.2  # Proportional gain
    Ki = 0.10  # Integral gain
    Kd = 0.008 # Derivative gain
    # Calculate the error
    error = setpoint - measured_value
    # Proportional term
    P = Kp * error
    
    # Integral term
    error_sum += error * dt
    I = Ki * error_sum
    
    # Derivative term
    D = Kd * (error - last_error) / dt
    
    # Calculate the control signal
    control_signal = P + I + D
    
    # Update the last error for the next iteration
    last_error = error
    
    return control_signal,error_sum, last_error


def get_current_pose (env):
    global_vel = env.engine.agent_manager.active_agents['default_agent'].velocity
    heading = env.engine.agent_manager.active_agents['default_agent'].heading_theta

    return global_vel, heading

def get_current_speed (env):
    speed = env.engine.agent_manager.active_agents['default_agent'].speed 
    return speed

def one_round_exp(env, lat_act, lon_act, base_speed, stable_time, collect_time_window_width, 
                plot_speed_before_stablization =  False, fit_visualization = False):
    # one round of experiment 
    env.reset()
    # here we use pid controller to stable the vehicle's lontitute speed to base speed
    speeds = []
    error_sum = 0 
    last_error = 0
    # print("experiment vars: lat action input, lon action input, base speed = ",  lat_act, lon_act, base_speed)
    for _ in range(int(stable_time * SAMPLING_FREQ)):
        global_vel, heading = get_current_pose(env)
        lon_speed = global_vel [0]*np.cos(heading) + global_vel[1]*np.sin(heading)

        action_lon, error_sum, last_error= pid_controller(base_speed,lon_speed, error_sum, last_error, 1/SAMPLING_FREQ)
        # print("action, error_sum, last_error = ", action_lon, error_sum, last_error)
        
        action = np.array([0, action_lon])
        o, r, d, info = env.step(action)
        
        speeds.append(get_current_speed(env))
    

    
    if plot_speed_before_stablization:
        plt.figure()
        # Plot time versus acceleration
        plt.plot(np.arange(len(speeds))/SAMPLING_FREQ, speeds, label='actual speed')
        plt.plot([0, len(speeds)/SAMPLING_FREQ], [base_speed, base_speed], label = "target speed")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('speed')
        plt.title('Time vs.speed')
        plt.show()
    

    # tests shows that the current PID params reached the speed specification
    # start collecting the data of given action
    vels = []
    headings = []
    for _ in range(int(collect_time_window_width * SAMPLING_FREQ)):
        action = np.array([lat_act, lon_act])
        
        
        o, r, d, info = env.step(action)
        vel, heading = get_current_pose(env)
        vels.append(vel)
        headings.append(heading)

    headings = np.array(headings)
    vels = np.array(vels)

    local_vel = get_local_from_heading(vels, headings) # vels and headings 
    ts = np.arange(int(collect_time_window_width * SAMPLING_FREQ))/SAMPLING_FREQ
    # local_acc = get_acc_from_vel(local_vel,ts, smooth_acc = True) 


    plot_slip_angle_gap =False
    if plot_slip_angle_gap:
        plt.figure()
        plt.plot(ts, np.arctan2(vels[:,1],vels[:,0]), label = 'vel dir')
        plt.plot(ts, headings, label = 'heading')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('heading (rad) and vel_direction (rad)')
        plt.show()
    
    # lon acc:
    lon_speed = local_vel[:,0]
    A = np.vstack([ts, np.ones(len(ts))])
    lon_acc, lon_offset = np.linalg.lstsq(A.T, lon_speed, rcond = None)[0]
    lon_sse = np.linalg.lstsq(A.T, lon_speed, rcond = None)[1][0]
    
    # lat acc:
    lat_speed = local_vel[:,1]
    lat_acc, lat_offset = np.linalg.lstsq(A.T, lat_speed, rcond = None)[0]
    lat_sse = np.linalg.lstsq(A.T, lat_speed, rcond = None)[1][0]

    if (lon_sse >= 3):
        lon_acc = (lon_speed[1] - lon_speed[0])*SAMPLING_FREQ
        lat_acc = (lat_speed[1] - lat_speed[0])*SAMPLING_FREQ
        lon_sse, lat_sse = -1, -1 # invalid SSE value 

    # speed visualization

    # fit_visualization = (lon_sse >= 3)
    if fit_visualization:
        plt.figure()
        # Plot time versus acceleration
        plt.scatter(ts, local_vel[:,0], label = 'local lat')
        plt.scatter(ts, local_vel[:,1], label = 'local lon')
        plt.plot([ ts[0],ts[-1]],[lat_acc * ts[0] + lat_offset, lat_acc * ts[-1] + lat_offset], label = "fitted lat")
        plt.plot([ ts[0],ts[-1]],[lon_acc * ts[0] + lon_offset, lon_acc * ts[-1] + lon_offset], label = "fitted lon")
        # plt.plot(ts, vels[:,0], ts, vels[:,1], label = 'global')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('local Velocity')
        plt.title(f"Time vs. Velocity(acc lon , acc_sse = {lon_acc: {3}.{3}},{lon_sse: {3}.{3}}")
        plt.show()

    return lat_acc, lon_acc, lat_sse, lon_sse 

def collect_action_acc_grid_data(args):
    
    '''
    sweep action space [-1,+1]* [-1, +1], with grain 10*10 in default,
    collect corresponding acc data

    '''
    
    num_dp_per_dim = args['num_dp_per_dim'] 
    data_dir = args['data_dir']

    # speeds= [1, 5, 10, 20] # min: 0 km/h, max: 72 km/h
    # speeds = [20]
    speeds = np.linspace(1,20,num_dp_per_dim)
    lats = np.linspace(-1,1,num_dp_per_dim) # in metadrive visualization, the steering bar, with +1 (left) -> -1 (right)
    lons = np.linspace(-1,1,num_dp_per_dim)*(-1) # in metadrive visualization
    # action_lat, action_lon = np.meshgrid(lats,lons)
    collect_window_width = 0.3 # sec
    stable_time = 10 # sec



    env = AddCostToRewardEnv_base(
    {
        "manual_control": False,
        "traffic_density": 0,
        
        # "case_num": num_dp_per_dim**2*len(speeds),
        "random_lane_width": False,
        "map_config": dict(
            type = 'block_num',
            config = 2,
            lane_width= 50
        ),
        "physics_world_step_size": 1/SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
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
    
    2. collect 0.5 sec (0.5*10 = 5 dps) of lontitute and latitute speed data when you maintain the specified throttle position, and steering angle.

    3. assume the base speed does not change during 0.5 sec, fit a line to the 5 data points collected. the slope would be the estimated acceleration 
    (we could also plot residual vesus time to see if the linearization approximation makes sense)

    4. record the (slope,std) into a num_dp_per_dim*num_dp_per_dim*num_dp_per_dim map 
        
    '''
    
   
    env.reset()
    log_data = [] # save as vars(lat action input, lon action input, base speed) + lat_acc + lon_acc
    for i in range(num_dp_per_dim): # action 0
        for j in range(num_dp_per_dim): # action 1
            for base_speed in speeds:
            # for base_speed in [20]:
                lat_acc, lon_acc, lat_sse, lon_sse = one_round_exp(env, lats[i], lons[j], base_speed, stable_time, collect_window_width, True, True)
                log = [lats[i], lons[j], base_speed, lat_acc, lon_acc, lat_sse, lon_sse]
                print("lat action input, lon action input, base speed, lat_acc, lon_acc, lat_sse, lon_sse= {:.{}f}, {:.{}f}, {:.{}f}, {:.{}f}, {:.{}f}, {:.{}f}, {:.{}f}".format(log[0], 3, log[1], 3, log[2], 3, log[3], 3, log[4], 3, log[5], 3, log[6], 3))
                log_data.append([lats[i], lons[j], base_speed, lat_acc, lon_acc, lat_sse, lon_sse])
                            
    log_data = np.array([log_data])
    np.save(data_dir,log_data)


    env.close()





if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--lambda', '-lam', type=float, default=1.)

    parser.add_argument('--num_dp_per_dim', '-num', type=float, default=10)
    parser.add_argument('--data_dir', type=str, default="examples/metadrive/map_action_to_acc/log/test.npy")
    args = parser.parse_args()
    args = vars(args)

    collect_action_acc_grid_data(args)