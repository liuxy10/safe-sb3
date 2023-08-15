
import gym
from gym import spaces
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np
from metadrive.component.vehicle_model.bicycle_model import BicycleModel



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


class AddCostToRewardEnv_base(MetaDriveEnv):
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


class BicycleModelEgoEnv_base(MetaDriveEnv):
    # In this BicycleModelEgoEnv_base, we overwrite the pybullet simulated dynamics to a simple bicycleModel
    # action space becomes [steer angle, acc]
    def step(self, actions):
        actions = self._preprocess_actions(actions)
        engine_info = self._step_simulator(actions)
        o, r, d, i = self._get_step_return(actions, engine_info=engine_info)
        return o, r, d, i


############## query map stuff #########################


def get_unique_vals(dat):
    # dat should be consistent with data collected from collect_action_acc_pair.py
    # dat in the form of lat action input, lon action input, base speed, lat_acc, lon_acc, lat_sse, lon_sse

    lat_acc_vals, lon_acc_vals, base_speed_vals = np.unique(dat[:,3]), np.unique(dat[:,4]), np.unique(dat[:,2])  
    return  lat_acc_vals, lon_acc_vals, base_speed_vals



def query_nearest_value_1d(query, vals):
    min_idx = np.argmin(np.abs(vals - query))
    return vals[min_idx]

def find_nearest_pt_2d_index(x,y, xs, ys):
    dist = (xs-x)**2 + (ys - y)**2
    return  np.argmin(dist)

def estimate_action(dat, query_speed, query_lat_acc, query_lon_acc, use_2nd_pt = False):
    # first, we take the nearest speed bin
    base_speed_vals = np.unique(dat[:,2])  
    speed = query_nearest_value_1d(query_speed, base_speed_vals)
    
    # then we 'slice' speed to attain acc space given a speed
    dat_lat_act, dat_lon_act,dat_base_speed, dat_lat_acc, dat_lon_acc, dat_lat_sse, dat_lon_sse = dat[:,0], dat[:,1], dat[:,2], dat[:,3], dat[:,4], dat[:,5], dat[:,6]
    lat_acc_given_speed = dat_lat_acc[dat_base_speed == speed]
    lon_acc_given_speed = dat_lon_acc[dat_base_speed == speed]
    lat_act_given_speed = dat_lat_act[dat_base_speed == speed]
    lon_act_given_speed = dat_lon_act[dat_base_speed == speed]


    if use_2nd_pt:
        ## TODO:  find 2 nearest points in acc space and linear intepolate
        pass

    else: 
        # Easiest way out: find the nearest point in acc space!
        idx = find_nearest_pt_2d_index(query_lat_acc, query_lon_acc, lat_acc_given_speed, lon_acc_given_speed)
        return np.array([lat_act_given_speed[idx], lon_act_given_speed[idx]])


    
################# some packaging utils ####################









################ some geometry utils ####################

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d




def get_local_from_heading(glb, headings):

    # global pose (vel) should be in the shape of (n,2), headings should contains headings (n) in the unit of radians

    return np.array([glb [:,0]*np.cos(headings) + glb [:,1]*np.sin(headings), 
                            glb [:,0]*np.sin(headings) - glb [:,1]*np.cos(headings)]).T

def calculate_diff_from_whole_trajectory(xs, ts, idx, n_f=30, n_b=30):
    # xs, ys, ts: traj in x, y axis
    # idx: idx that we want to estimate the 1st diff
    # n_f, n_b: data points at front, at back (eg. idx -n_f |____|____| idx + n_f)
    # Interpolate the data using a quadratic spline
    tck_x = interp1d(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], 
                     xs[max(0, idx - n_f): min(ts.shape[0], idx + n_b)], kind='cubic', bounds_error=False, fill_value="extrapolate")
    dt = np.mean(np.diff(ts))
    # Differentiate the splines to obtain the velocity in the x and y directions
    dxs = np.gradient(tck_x(ts[max(0, idx - n_f): min(ts.shape[0], idx + n_b)]), dt)
    
    dx= dxs[min(n_f-1, idx)]
    return dx

def get_acc_from_vel(velocity,ts, smooth_acc = False):
    acc = np.zeros_like(velocity)
    acc[:,0] = [calculate_diff_from_whole_trajectory(velocity[:,0], ts, i) for i in range(velocity.shape[0])] 
    acc[:,1] = [calculate_diff_from_whole_trajectory(velocity[:,1], ts, i) for i in range(velocity.shape[0])] 

    if smooth_acc:

        acc[:,0] = savgol_filter(acc[:,0], 20, 3)
        acc[:,1] = savgol_filter(acc[:,1], 20, 3)

    return acc


def get_acc_from_speed(speed,ts, smooth_acc = False):
    acc = np.array([calculate_diff_from_whole_trajectory(speed, ts, i) for i in range(speed.shape[0])] )
    if smooth_acc:
        acc = savgol_filter(acc, 20, 3)

    return acc

def get_rate_from_heading(heading,ts, smooth_rate = False):
    MAX_HEADING_RATE = np.pi/2 # random 
    dt = 0.1
    heading = get_continuous_heading(heading, dt, MAX_HEADING_RATE)
    rate = np.array([calculate_diff_from_whole_trajectory(heading, ts, i) for i in range(heading.shape[0])] )
    
    if smooth_rate:
        rate = savgol_filter(rate, 20, 3)

    return rate

def get_continuous_heading(heading, dt, max_hr):
    for i in range(heading.shape[0]):
        h = heading[i]
        if i < heading.shape[0] -1:
            if abs(h - heading[i+1]) > max_hr * dt:
                assert np.min((h%np.pi, -h % np.pi)) < max_hr * dt, "max_hr is set too small"
                heading[i+1:] += np.pi * 2 * np.sign(h - heading[i+1])
    return heading

    


########### debug #####################
def check_observation_action_space(env):
    # Access the observation space
    observation_space = env.observation_space

    # Print information about the observation space
    print("Observation space:", observation_space)
    print("Observation space shape:", observation_space.shape)
    print("Observation space high:", observation_space.high)
    print("Observation space low:", observation_space.low)

    # Access the action space
    action_space = env.action_space

    # Print information about the action space
    print("Action space:", action_space)
    print("Action space shape:", action_space.shape)
    print("Action space high:", action_space.high)
    print("Action space low:", action_space.low)




# Define the mapping function from discrete to continuous actions
def map_discrete_to_continuous(value, num_values, lb, hb):
    return [(value[i]/num_values)*(hb-lb) + lb for i in range(len(value))]

# Define the mapping function from continuous to discrete values
def map_continuous_to_discrete(value,num_values, lb, hb):
    return [int((value[i] - lb) * (num_values) / (hb-lb)) for i in range(len(value))]

if __name__ == "__main__":
    # Define the number of discrete values for each dimension
    num_values = 100

    # Define the continuous value space
    continuous_value_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    
    # Define the discrete value space
    discrete_value_space = spaces.MultiDiscrete([num_values] * continuous_value_space.shape[0])
    # Print information about the observation space
    print("discrete space:", discrete_value_space)
    print("discrete space shape:", discrete_value_space.shape)
    # Test the discretization
    continuous_value = [0.54, -0.76]
    discrete_value = map_continuous_to_discrete(continuous_value, num_values, -1.0, 1.0)
    continuous_value_restored = map_discrete_to_continuous(discrete_value, num_values, -1.0, 1.0)

    print("Continuous value:", continuous_value)
    print("Discrete value:", discrete_value)
    print("Restored Continuous value:", continuous_value_restored)