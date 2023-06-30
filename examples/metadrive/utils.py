
import gym
from gym import spaces
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np



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

################ some geometry utils ####################

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

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

def get_global_acc(velocity,ts, smooth_acc = False):
    acc = np.zeros_like(velocity)
    acc[:,0] = [calculate_diff_from_whole_trajectory(velocity[:,0], ts, i) for i in range(velocity.shape[0])] 
    acc[:,1] = [calculate_diff_from_whole_trajectory(velocity[:,1], ts, i) for i in range(velocity.shape[0])] 

    if smooth_acc:

        acc[:,0] = savgol_filter(acc[:,0], 20, 3)
        acc[:,1] = savgol_filter(acc[:,1], 20, 3)

    return acc












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