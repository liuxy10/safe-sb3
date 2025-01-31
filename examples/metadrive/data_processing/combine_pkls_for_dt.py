
import sys
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
from utils import AddCostToRewardEnv, get_acc_from_vel, get_local_from_heading, get_acc_from_speed, get_rate_from_heading
import numpy as np

from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from collect_h5py_from_pkl import get_current_ego_trajectory_old
# from metadrive.utils.coordinates_shift import waymo_2_metadrive_heading, waymo_2_metadrive_position


import tqdm
import pickle
import h5py
import os
import matplotlib.pyplot as plt
import re




WAYMO_SAMPLING_FREQ = 10
TOTAL_TIMESTAMP = 90


def extract_numbers_from_filename(filename):
    # Use regular expression to find all numeric sequences in the filename
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]


def save_data_to_pickle(filename, data):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
            existing_data.append(data)
        except:
            print(f"skipping data{filename}")
    else:
        existing_data = [data]

    with open(filename, 'wb') as f:
        pickle.dump(existing_data, f)


def pkls_to_h5py_for_bc(pkl_dir, h5py_path):

    pkl_list = os.listdir(pkl_dir)

    trajectories = []
    for pkl_fn in pkl_list:
        with open(os.path.join(pkl_dir, pkl_fn), 'rb') as f:
            try:
                temp = pickle.load(f)
                trajectories.extend(temp)
            except EOFError:
                print("........ skipping " + pkl_fn + " ........")

    trajectories = sorted(trajectories, key=lambda x: x['seed'])

    traj_flatten = {}
    f = h5py.File(h5py_path, 'w')

    for name in trajectories[0].keys():
        data = np.array([path[name] for path in trajectories])
        if name in ["rewards", "dones"]:
            data = data.reshape(-1, 1)
            traj_flatten[name] = data.reshape(-1, 1)
        traj_flatten[name] = np.vstack(data)
        print("flatten traj for ", name, traj_flatten[name].shape)

    for name in traj_flatten.keys():
        data = traj_flatten[name]
        if len(data.shape) == 2:
            print("Saving ", name)
            f.create_dataset(name, (data.shape[0], data.shape[1]), data=data)
        else:
            print("Saving ", name)
            print(name, data, data.shape)
            f.create_dataset(name, (data.shape[0], ), data=data)

    f.close()


def check_start_seed(filename):

    params = extract_numbers_from_filename(filename)
    default_start_seed = 0
    if len(params) == 3:
        default_start_seed = params[-1]*2000
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
            return existing_data[-1]["seed"]
        except:
            print("numpy corrupted, reload")
            return default_start_seed
    else:
        return default_start_seed


def collect_rollout_in_one_seed(env, seed):
    obs_rec = np.ndarray((0, ) + env.observation_space.shape)
    ac_rec = np.ndarray((0, ) + env.action_space.shape)
    re_rec = np.ndarray((0, ))
    terminal_rec = np.ndarray((0, ), dtype=bool)
    cost_rec = np.ndarray((0, ))

    env.reset(force_seed=seed)
    # ts, _, vel, _ = get_current_ego_trajectory(env,seed)
    ts, _, vel, acc, heading, heading_rate = get_current_ego_trajectory_old(
        env, seed)
    N = acc.shape[0]
    speed = np.linalg.norm(vel, axis=1)
    # for t in tqdm.trange(N, desc="Timestep"):
    for t in range(N):
        # action = np.array([heading_rate[t], acc[t]])
        action = np.array([0, 0])

        # whatever the input action is overwrited to be zero (due to the replay policy)
        obs, reward, done, info = env.step(action)
        # if reward < 0:
        # assume info['out_of_road'] == True,
        # print(reward, info['step_reward'], info['cost'], info['out_of_road'])
        obs_rec = np.concatenate((obs_rec, obs.reshape(1, obs.shape[0])))
        ac_rec = np.concatenate((ac_rec, action.reshape(1, action.shape[0])))
        re_rec = np.concatenate((re_rec, np.array([reward])))
        terminal_rec = np.concatenate((terminal_rec, np.array([done])))
        cost_rec = np.concatenate((cost_rec, np.array([info['cost']])))

    print("seed, max speed, avg speed, acc range,  heading range, reward range, cost range = {:.{}f}, {:.{}f}, {:.{}f}, [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}], [{:.{}f}, {:.{}f}]".format(seed, 1,
                                                                                                                                                                                                               np.max(vel), 3, np.mean(vel), 3, np.min(acc), 3, np.max(acc), 3, np.min(heading), 3, np.max(heading), 3, np.min(re_rec), 3, np.max(re_rec), 3, np.min(cost_rec), 3, np.max(cost_rec), 3))
    data = {
        "observations": obs_rec[:N-1],
        "next_observations": obs_rec[1:N],
        "actions": ac_rec[:N-1],
        "rewards": re_rec[:N-1],
        "dones": terminal_rec[:N-1],
        "costs": cost_rec[:N-1],
        "seed": seed

    }
    return data


def main(args):
    file_list = os.listdir(args['pkl_dir'])
    if args['num_of_scenarios'] == 'ALL':
        num_scenarios = len(file_list)
    else:
        num_scenarios = int(args['num_of_scenarios'])
        # num_scenarios = 10
    print("num of scenarios: ", num_scenarios)

    # check starting scenarios:
    start_seed = args['start_seed']
    print("----------------start_seed = "+str(start_seed)+"!------------")
    env = AddCostToRewardEnv(
        {
            "manual_control": False,
            "no_traffic": False,
            "agent_policy": ReplayEgoCarPolicy,
            "waymo_data_directory": args['pkl_dir'],
            "case_num":args["num_of_scenarios"],
            "start_seed": args["start_seed"],
            # "evironment_num": args["num_of_scenarios"],
            # have to be specified each time we use waymo environment for training purpose
            "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ,
            "use_render": False,
            "reactive_traffic": False,
            # "vehicle_config": dict(
            #     show_lidar=True,
            #     # no_wheel_friction=True,
            #     lidar=dict(num_lasers=0))
            "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=80, distance=50, num_others=4),  # 120
                lane_line_detector=dict(num_lasers=12, distance=50),  # 12
                side_detector=dict(num_lasers=20, distance=50)  # 160
            )
        }, lamb=args["lamb"]
    )

    # init all recorded variables

    data = []

    # for seed in range(start_seed, start_seed + num_scenarios):
    for seed in tqdm.trange(num_scenarios, desc="seeds"):
        seed += start_seed
        # import pdb; pdb.set_trace()
        # print(seed)

        data = collect_rollout_in_one_seed(env, seed)
        save_data_to_pickle(args['dt_data_path'], data)

    env.close()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pkl_dir', type=str, default='/home/vision/src/data/metadrive/pkl_9/')
    parser.add_argument('--pkl_dir', type=str,
                        default='~/src/data/metadrive/pkl_9/')
    parser.add_argument('--start_seed', type=int, default='0')
    parser.add_argument('--dt_data_path', type=str,
                        default='~/src/data/metadrive/dt_pkl/waymo_n_10000_lam_1_eps_10/eps_0.pkl')
    parser.add_argument('--num_of_scenarios', type=int, default='1000')
    parser.add_argument('--lamb', type=float, default=1.)
    # parser.add_argument('--map_dir', type = str, default = 'examples/metadrive/map_action_to_acc/log/test.npy')
    args = parser.parse_args()
    args = vars(args)

    main(args)

    # dt_pkl_dir = '/home/xinyi/src/data/metadrive/dt_pkl/waymo_n_10000_lam_1_eps_10'
    # h5py_path = '/home/xinyi/src/data/metadrive/h5py/waymo_n_10000_lam_1.h5py'
    # pkls_to_h5py_for_bc(dt_pkl_dir, h5py_path)
