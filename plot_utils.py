import numpy as np
import matplotlib.pyplot as plt

def plot_distances(goal_distances, estimated_obstacle_distances, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt

    obst_radius = 0.6

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(time_steps, goal_distances, label='Distance to Goal')

    num_obstacles = estimated_obstacle_distances.shape[1]
    for i in range(num_obstacles):
        plt.plot(time_steps, estimated_obstacle_distances[:, i] - obst_radius, label=f'Estimated Distance to Obstacle {i+1}')

    plt.axhline(y=0.2, color='black', linestyle='--', linewidth=3, label='Safety Margin')

    plt.xlabel('Time (s)')
    plt.ylabel('Distance')
    plt.title('Distances to Goal and Estimated Distances to Obstacles over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)