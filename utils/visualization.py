import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_trajectory(history):
    """
    history: a numpy array of shape (N, 6) containing [x, y, z, vx, vy, vz]
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_cog(111, projection='3d')

    # Extract coordinates
    x, y, z = history[:, 0], history[:, 1], history[:, 2]

    # Plot the Target (ISS) at the center
    ax.scatter(0, 0, 0, color='red', s=100, label='Target (ISS)')
    
    # Plot the Chaser's path
    ax.plot(x, y, z, label='Chaser Path', color='blue', alpha=0.7)
    
    # Labeling the axes (Hill's Frame)
    ax.set_xlabel('Radial (x)')
    ax.set_ylabel('Along-Track (y)')
    ax.set_zlabel('Cross-Track (z)')
    ax.set_title('Spacecraft Docking Trajectory')
    ax.legend()
    plt.show()