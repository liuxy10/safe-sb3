# visualize map

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/xinyi/Documents/UCB/safe-sb3/examples/metadrive")
from utils import get_acc_from_vel, get_local_from_heading, estimate_action, query_nearest_value_1d

def plot_action_as_variable(dat):
   

    lat_act, lon_act, base_speed, lat_acc, lon_acc, lat_sse, lon_sse = dat.T[:]
     # Create a 3D plot
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    
    for speed in np.unique(base_speed):
        
        ax1.scatter(lat_act[base_speed == speed], lon_act[base_speed == speed], lat_acc[base_speed == speed], label = "speed = "+str(speed))
       

    # Set labels and title
    ax1.set_xlabel('Latitude action')
    ax1.set_ylabel('Longitude action')
    ax1.set_zlabel('Latitude acceleration')
    ax1.set_title('latitude acceleration under different base speed, lat action, lon action')
    ax1.legend()

    # FOR LON ACC, COPY THE ABOVE CODE

    for speed in np.unique(base_speed):
        ax2.scatter(lat_act[base_speed == speed], lon_act[base_speed == speed], lon_acc[base_speed == speed], label = "speed = "+str(speed))
        
    # Set labels and title
    ax2.set_xlabel('Latitude action')
    ax2.set_ylabel('Longitude action')
    ax2.set_zlabel('Latitude acceleration')
    ax2.set_title('Lontitute acceleration under different base speed, lat action, lon action')
    ax2.legend()

    plt.show()
    # plt.savefig(fig, "examples/metadrive/map_action_to_acc/log/visualize_map.png")

# one way is to visualize acceleration surfaces of different base speeds on the actions dimensions

# however, in simulation, we will acquire speed and acc_lat. acc_lon, 
# we want to know if there is a unique mapping from (speed, acc_lat, acc_lon) -> (act_lat, act_lon)


def plot_acceleration_as_variable(dat, query = None):

    lat_act, lon_act, base_speed, lat_acc, lon_acc, lat_sse, lon_sse = dat.T[:]
     # Create a 3D plot
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    plot_speed_range = [1,20]

    if query == None:
        pass
    else:
        query_speed, query_lat_acc, query_lon_acc = query[0], query[1], query[2]
        est_act = estimate_action(dat, query_speed, query_lat_acc, query_lon_acc)
        speed_given = query_nearest_value_1d(query[0], np.unique(base_speed))
        plot_speed_range = [speed_given]
        ax1.scatter(query_lat_acc, query_lon_acc, est_act[0], label = 'action estimate')
        ax2.scatter(query_lat_acc, query_lon_acc, est_act[1], label = 'action estimate')

    
    # for speed in np.unique(base_speed):
    for speed in plot_speed_range:
        
        ax1.scatter(lat_acc[base_speed == speed], lon_acc[base_speed == speed], lat_act[base_speed == speed],  label = f"speed = {speed: {3}.{3}}")
       

    # Set labels and title
    ax1.set_xlabel('Latitude acceleration')
    ax1.set_ylabel('Longitude acceleration')
    ax1.set_zlabel('Latitude action')
    ax1.set_title('Latitute action inferred from different base speed, lat acc, lon acc')
    ax1.legend()

    # FOR LON ACC, COPY THE ABOVE CODE

    for speed in plot_speed_range: #np.unique(base_speed):
        ax2.scatter(lat_acc[base_speed == speed], lon_acc[base_speed == speed], lon_act[base_speed == speed], label = f"speed = {speed: {3}.{3}}")
        
    # Set labels and title
    ax2.set_xlabel('Latitude acceleration')
    ax2.set_ylabel('Longitude acceleration')
    ax2.set_zlabel('Lontitute action')
    ax2.set_title('Lontitute action inferred from different base speed, lat acc, lon acc')
    ax2.legend()

    plt.show()


    



def plot_reachable_region(base_speed, lat_acc, lon_acc, query = None):

     # Create a 3D plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d') 
    ax1.scatter(lat_acc, lon_acc, base_speed, label = "data")

    if query == None:
        pass
    else:
        query_speed, query_lat_acc, query_lon_acc = query[0], query[1], query[2]
        ax1.scatter(query_lat_acc, query_lon_acc , query_speed, label = 'query')

       

    # Set labels and title
    ax1.set_xlabel('Latitude acceleration')
    ax1.set_ylabel('Longitude acceleration')
    ax1.set_zlabel('Base_speed')
    ax1.set_title('The reachable region')
    ax1.legend()

    plt.show()




if __name__ == "__main__":

    # lat action input, lon action input, base speed + lat_acc + lon_acc + 
    dat = np.load("examples/metadrive/map_action_to_acc/log/test.npy")[0]
    lat_act, lon_act, base_speed, lat_acc, lon_acc, lat_sse, lon_sse = dat.T[:]
    print(dat.shape)

    query = [10, 1, 5]
    # plot_acceleration_as_variable(dat, query)
    plot_reachable_region(base_speed, lat_acc, lon_acc, query)

    # test the query function 
    # estimate_action(dat, )



    print(dat.shape)


