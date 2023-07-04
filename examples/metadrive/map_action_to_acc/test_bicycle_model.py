import numpy as np

from metadrive.component.vehicle_model.bicycle_model import BicycleModel
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger
from metadrive.utils.math_utils import norm
import matplotlib.pyplot as plt


def predict(current_state, actions, model):
    model.reset(*current_state)
    for action in actions:
        model.predict(0.1, action)
    return model.state


def _test_bicycle_model():
    horizon = 10
    action_last = 20
    setup_logger(True)
    env = MetaDriveEnv(
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
        "physics_world_step_size": 1/10, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
    }
    )

    bicycle_model = BicycleModel()
    o = env.reset()
    vehicle = env.current_track_vehicle
    v_dir = vehicle.velocity_direction
    bicycle_model.reset(*vehicle.position, vehicle.speed, vehicle.heading_theta, np.arctan2(v_dir[1], v_dir[0]))
    actions = []
    for steering in [1.0, 0.8, 0.6, 0.4, 0.2, 0]:
        for dir in [-1, 1]:
            s = dir * steering
            for throttle in [1.0, 0.8, 0.6, 0.4, 0.2, 0, -0.5]:
                actions += [[s, throttle]] * action_last
    predict_states = []
    actual_pos = []
    predict_pos = []
    for s in range(len(actions)):
        vehicle = env.current_track_vehicle
        v_dir = vehicle.velocity_direction
        predict_states.append(
            predict(
                current_state=(
                    *env.current_track_vehicle.position, env.current_track_vehicle.speed,
                    env.current_track_vehicle.heading_theta, np.arctan2(v_dir[1], v_dir[0])
                ),
                actions=[actions[i] for i in range(s, min(s + horizon, len(actions)))],
                model=bicycle_model
            )
        )
        o, r, d, info = env.step(actions[s])
        actual_pos.append([vehicle.position[0],vehicle.position[1]])
        index = s - horizon
        if index >= 0:
            state = predict_states[index]
            print(norm(state["x"] - vehicle.position[0], state["y"] - vehicle.position[1]))

    actual_pos = np.array(actual_pos)
    predict_pos= np.array([[ps["x"], ps["y"]] for ps in predict_states])  
    plt.figure()
    plt.plot(actual_pos[:,0], actual_pos[:,1], label = "actual")
    plt.scatter(actual_pos[:,0], actual_pos[:,1])
    plt.plot(predict_pos[:,0], predict_pos[:,1], label = "predict")
    plt.legend()
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.show()
    print("done")
    




if __name__ == "__main__":
    _test_bicycle_model()
