# visualize map

import numpy as np
import matplotlib.pyplot as plt

def plot_with_log_data(dat):
    lat_act, lon_act, base_speed, lat_acc, lon_acc, lat_sse, lon_sse = dat.T[:]
     # Create a 3D plot
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    for speed in np.unique(base_speed):
        # one way to visualize: Plot the data using scatter
        ax1.scatter(lat_act[base_speed == speed], lon_act[base_speed == speed], lat_acc[base_speed == speed], label = "speed = "+str(speed))

    # Set labels and title
    ax1.set_xlabel('Latitude action')
    ax1.set_ylabel('Longitude action')
    ax1.set_zlabel('Latitude acceleration')
    ax1.set_title('latitude acceleration under different base speed, lat action, lon action')
    ax1.legend()

    # FOR LON ACC, COPY THE ABOVE CODE

    for speed in np.unique(base_speed):
        # one way to visualize: Plot the data using scatter
        ax2.scatter(lat_act[base_speed == speed], lon_act[base_speed == speed], lon_acc[base_speed == speed], label = "speed = "+str(speed))
         # another way to visualize: mesh plot
         # Create meshgrid
        
        # X, Y = np.meshgrid(lat_act[base_speed == speed], lon_act[base_speed == speed])
        # Z = np.zeros_like(X)
        # for i in range(Z.shape[0]):
        #     for j in range(Z.shape[1]):
        #         Z[i,j] = lat_acc[(base_speed == speed) & (lat_act == lat_act[base_speed == speed][i]) & (lon_act == lon_act[base_speed == speed][j])][0]
        # # Plot the surface
        # surf = ax.plot_surface(X, Y, Z, cmap='viridis',label = "speed = "+str(speed))
        

    # Set labels and title
    ax2.set_xlabel('Latitude action')
    ax2.set_ylabel('Longitude action')
    ax2.set_zlabel('Latitude acceleration')
    ax2.set_title('Lontitute acceleration under different base speed, lat action, lon action')
    ax2.legend()

    plt.show()
    plt.savefig(fig, "examples/metadrive/map_action_to_acc/log/visualize_map.png")

    print(lat_act.shape)
if __name__ == "__main__":

# lat action input, lon action input, base speed + lat_acc + lon_acc + 
    dat = np.load("examples/metadrive/map_action_to_acc/log/test.npy")[0]
    
    plot_with_log_data(dat)

    print(dat.shape)


