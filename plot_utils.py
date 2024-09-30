import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils_3d import *
from robot_config import *

def plot_distances(goal_distances, estimated_obstacle_distances, obst_radius, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot distance to goal as blue dotted line
    plt.plot(time_steps, goal_distances, color='red', linestyle=':', linewidth=3, label='Distance to Goal')

    # Remove the extra dimension if it exists
    if estimated_obstacle_distances.ndim == 3:
        estimated_obstacle_distances = estimated_obstacle_distances.squeeze(1)
    
    num_obstacles = estimated_obstacle_distances.shape[1]


    for i in range(num_obstacles):
        if i == 0:
            plt.plot(time_steps, estimated_obstacle_distances[:, i] - obst_radius, linewidth=3, label='Estimated Distance to Obstacles')
        else:
            plt.plot(time_steps, estimated_obstacle_distances[:, i] - obst_radius, linewidth=3)
    
    plt.axhline(y=0.05, color='black', linestyle='--', linewidth=3, label='Safety Margin')
    
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Distance', fontsize=18)
    # plt.title('Distances to Goal and Estimated Distances to Obstacles over Time')

    # Set tick label size to 16
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(fontsize=18, loc='upper right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# def sample_points_on_obstacle(obstacle, num_points_per_face):
#     sampled_points = []
#     for face in obstacle:
#         # Calculate the min and max coordinates of the face
#         min_coords = np.min(face, axis=0)
#         max_coords = np.max(face, axis=0)

#         # Determine the grid size based on the number of points required
#         num_points_sqrt = int(np.ceil(np.sqrt(num_points_per_face)))
#         x_grid = np.linspace(min_coords[0], max_coords[0], num_points_sqrt)
#         y_grid = np.linspace(min_coords[1], max_coords[1], num_points_sqrt)
#         z_grid = np.linspace(min_coords[2], max_coords[2], num_points_sqrt)

#         # Generate grid points and filter out those outside the face
#         grid_points = np.array(np.meshgrid(x_grid, y_grid, z_grid)).T.reshape(-1, 3)
#         grid_points = grid_points[
#             np.all(grid_points >= min_coords, axis=1) & np.all(grid_points <= max_coords, axis=1)
#         ]

#         # Sample the required number of points from the grid
#         # num_points = min(num_points_per_face, len(grid_points))
#         # idx = np.random.choice(len(grid_points), num_points, replace=False)

#         sampled_points.append(grid_points)

#     return np.concatenate(sampled_points, axis=0)

def sample_points_on_face(face, num_points_per_unit_area):
    # Calculate the min and max coordinates of the face
    min_coords = np.min(face, axis=0)
    max_coords = np.max(face, axis=0)


    # Calculate the dimensions of the face
    dimensions = max_coords - min_coords

    # Identify non-zero dimensions to calculate area correctly
    non_zero_dims = dimensions != 0

    # Calculate the area of the face
    area = np.prod(dimensions[non_zero_dims])

    # Calculate the number of points to sample based on the face area
    num_points_per_face = int(num_points_per_unit_area * area)

    # Generate random points within the face boundaries
    points = np.random.rand(num_points_per_face, 3)
    points[:, 0] = points[:, 0] * dimensions[0] + min_coords[0]
    points[:, 1] = points[:, 1] * dimensions[1] + min_coords[1]
    points[:, 2] = points[:, 2] * dimensions[2] + min_coords[2]

    return points


def generate_realistic_env_3d(corridor_pos, corridor_size, obstacle_positions, obstacle_sizes, num_points_per_unit_area=3):


    # Generate corridor walls (unchanged from original)
    wall_positions = [
        # Bottom face
        np.array([
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        ]),
        # Top face
        np.array([
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        ]),
        # Front face
        np.array([
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        ]),
        # Back face
        np.array([
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
            [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
            [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        ]),
    ]
    

    obstacle_shapes = []
    obstacle_points = []
    for pos, size in zip(obstacle_positions, obstacle_sizes):
        # Define obstacle as a collection of 6 faces
        obstacle = [
            # Bottom face
            np.array([
                [pos[0] - size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
            ]),
            # Top face
            np.array([
                [pos[0] - size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
            ]),
            # Front face
            np.array([
                [pos[0] - size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
            ]),
            # Back face
            np.array([
                [pos[0] - size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
            ]),
            # Left face
            np.array([
                [pos[0] - size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] - size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
            ]),
            # Right face
            np.array([
                [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
                [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
            ]),
        ]
        obstacle_shapes.append(obstacle)

        # Sample points on each face of the obstacle
        sampled_points = []
        for face in obstacle:
            face_points = sample_points_on_face(face, num_points_per_unit_area)
            
            sampled_points.append(face_points)

        # Concatenate the sampled points for the current obstacle
        obstacle_points.append(np.concatenate(sampled_points, axis=0))

    # Concatenate all obstacle points into a single array
    obstacle_points = np.concatenate(obstacle_points, axis=0)

    return wall_positions, obstacle_shapes, obstacle_points

def generate_sphere_points(sphere_positions, obst_radius=0.3, num_points_per_face=30):
    sphere_points = []
    for sphere_pos in sphere_positions:
        # Generate points on the sphere surface
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obst_radius * np.cos(u) * np.sin(v) + sphere_pos[0]
        y = obst_radius * np.sin(u) * np.sin(v) + sphere_pos[1]
        z = obst_radius * np.cos(v) + sphere_pos[2]
        sphere_points_temp = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        # Randomly select points from the sphere surface
        num_points = min(num_points_per_face, len(sphere_points_temp))
        idx = np.random.choice(len(sphere_points_temp), num_points, replace=False)
        sampled_sphere_points = sphere_points_temp[idx]

        sphere_points.append(sampled_sphere_points)

    sphere_points = np.concatenate(sphere_points, axis=0)
    return sphere_points

def plot_env_3d(wall_positions, obstacle_shapes, goal_position, sphere_positions, sphere_radius, ax, plt_show = False):

    obstacle_plots = []
    for i, sphere_pos in enumerate(sphere_positions):
        # Create a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = sphere_radius * np.cos(u) * np.sin(v) + sphere_pos[0]
        y = sphere_radius * np.sin(u) * np.sin(v) + sphere_pos[1]
        z = sphere_radius * np.cos(v) + sphere_pos[2]

        # Plot the sphere
        sphere = ax.plot_surface(x, y, z, color='red', alpha=0.6)

        if i == 0:
            obstacle_plot = ax.scatter([sphere_pos[0]], [sphere_pos[1]], [sphere_pos[2]], color='red', s=100, label='Obstacles')
            obstacle_plots.append(obstacle_plot)

    # Plot  walls
    for wall in wall_positions:
        ax.add_collection3d(Poly3DCollection([wall], facecolors='black', linewidths=1, edgecolors='k', alpha=0.2))

    # Plot obstacles with different colors
    colors = ['red', 'green',  'orange', 'blue']
    
    for obstacle, color in zip(obstacle_shapes, colors):
        for face in obstacle:
            ax.add_collection3d(Poly3DCollection([face], facecolors=color, linewidths=1, edgecolors='k', alpha=0.3))

    # ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], c='black', s=5)

    # Plot goal position
    ax.scatter(*goal_position, c='blue', marker='*', s=500)

    plt.tight_layout()

    # Remove axis lines, ticks, and numbers
    ax.set_axis_off()

    # Remove background grid
    ax.grid(False)

    # Remove background panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    if plt_show:
        plt.show()


def generate_random_env_3d(num_obstacles, xlim, ylim, zlim, goal_xlim, goal_ylim, goal_zlim, min_distance_obs, min_distance_goal):
    # Generate random obstacle positions
    obstacle_positions = []
    obstacle_velocities = []

    for _ in range(num_obstacles):
        while True:
            pos = np.random.uniform(low=[xlim[0], ylim[0], zlim[0]], high=[xlim[1], ylim[1], zlim[1]])
            if all(np.linalg.norm(pos - np.array(obs_pos)) >= min_distance_obs for obs_pos in obstacle_positions):
                obstacle_positions.append(pos)
                break
        vel = np.random.rand(3)

        obstacle_velocities.append(vel)

    obstacle_positions = np.array(obstacle_positions)

    obstacle_velocities = np.array(obstacle_velocities)

    # Generate random goal point
    while True:
        goal_point = np.random.uniform(low=[goal_xlim[0], goal_ylim[0], goal_zlim[0]], high=[goal_xlim[1], goal_ylim[1], goal_zlim[1]])
        if all(np.linalg.norm(goal_point - obs_pos) >= min_distance_goal for obs_pos in obstacle_positions):
            break


    # Hard-code obstacle positions and velocities
    # obstacle_positions = [
    #     np.array([1., 1., 2.]),  # First obstacle
    #     np.array([1., 1., 4.]),   # Second obstacle
    #     np.array([2., 2., 3.]),  # First obstacle
    #     np.array([3., 1., 4.]),   # Second obstacle

    # ]


    # obstacle_positions = np.array(obstacle_positions)

    goal_point = np.array([4., 0., 4.])

    return obstacle_positions, obstacle_velocities, goal_point



def plot_random_env_3d(obstacle_positions, goal_point, obst_radius, ax, plt_show=False):
    # Plot spherical obstacles
    for sphere_pos in obstacle_positions:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obst_radius * np.cos(u) * np.sin(v) + sphere_pos[0]
        y = obst_radius * np.sin(u) * np.sin(v) + sphere_pos[1]
        z = obst_radius * np.cos(v) + sphere_pos[2]
        ax.plot_surface(x, y, z, color='red', alpha=0.6)

    # Plot goal position
    ax.scatter(*goal_point, c='blue', marker='*', s=500)

    # Remove axis lines, ticks, and numbers
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    if plt_show:
        plt.show()


def save_high_res_figure(fig, filename, dpi=300):
    os.makedirs('figures', exist_ok=True)
    fig.savefig(f'figures/{filename}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)



def plot_and_save_realsitic_env():

    # Define the environment parameters

    corridor_pos = np.array([0, 0, -1])

    corridor_size = np.array([2, 2, 2])

    obstacle_positions = [
        np.array([2.5, 0, 4.8]),
        np.array([2.5, 0, 0]),
        #np.array([1, -4, 3]), 
        np.array([5.3, 0, 2])
    ]
    obstacle_sizes = [
        np.array([1, 5, 3.2]),
        np.array([1, 5, 2]),
        #np.array([3, 2, 4]),
        np.array([2., 5, 1])
    ]


    goal_position = np.array([4.8, 0, 3.6])

    # Generate dynamic spheres
    sphere_positions = np.array([
        np.array([4.0, 0.0, 3.5]),  # Sphere 1 position
        np.array([4.0, 0.0, 5.0])  # Sphere 2 position
    ])
    sphere_velocities = np.array([
        np.array([0.0, 1.0, 0.0]),  # Sphere 1 velocity
        np.array([0.0, 0.0, 1.0])  # Sphere 2 velocity
    ])

    
    sphere_radius = 0.5

    wall_positions, obstacle_shapes, obstacle_points = generate_realistic_env_3d(
        corridor_pos, corridor_size, obstacle_positions, obstacle_sizes)
    
    sphere_points = generate_sphere_points(sphere_positions, obst_radius=sphere_radius)

    obstacle_points = np.concatenate((obstacle_points, sphere_points), axis=0)

    
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    ax.set_xlim(-1, 7)
    ax.set_ylim(-4, 4)
    ax.set_zlim(0, 8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    # cable_lengths = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.], [2.0, 2.0], [2.0, 2.0]])

    # plot a specific configuration
    cable_lengths = jnp.array([[2.0, 2.1], [2.1, 2.0], [1.9, 2.], [1.9, 1.9], [2.0, 2.0]])

    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH

    plot_links_3d(cable_lengths, link_radius, link_length, ax, base_center, base_normal)

    plot_env_3d(wall_positions, obstacle_shapes, goal_position, sphere_positions, sphere_radius, ax, plt_show = True)

    # save_high_res_figure(fig, '3d_environment.png')

    # Close the figure to free up memory
    plt.close(fig)

def plot_and_save_random_env():

    xlim = [-4, 4]
    ylim = [-4, 4]
    zlim = [-1, 6.0]

    goal_xlim = [-3.0, 3.0]
    goal_ylim = [-3.0, 3.0]
    goal_zlim = [0.0, 5.0]

    min_distance_obs = 1.5
    min_distance_goal = 2.0

    obst_radius = 0.6

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    cable_lengths = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.], [2.0, 2.0]])

    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH


    obstacle_positions, obstacle_velocities, goal_point = generate_random_env_3d(
        num_obstacles=12, xlim=xlim, ylim=ylim, zlim=zlim,
        goal_xlim=goal_xlim, goal_ylim=goal_ylim, goal_zlim=goal_zlim,
        min_distance_obs=min_distance_obs, min_distance_goal=min_distance_goal)
    
    plot_links_3d(cable_lengths, link_radius, link_length, ax, base_center, base_normal)
    
    # Plot the random environment
    plot_random_env_3d(obstacle_positions, goal_point, obst_radius, ax, plt_show=True)
    
    save_high_res_figure(fig, '3d_random_environment.png')

    plt.close(fig)



if __name__ == '__main__':

    # plot the realistic env
    plot_and_save_realsitic_env()

    # plot a random env with multiple dynamic sphere obstacles
    plot_and_save_random_env()







