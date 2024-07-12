import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils_3d import *
from robot_config import *

def plot_distances(goal_distances, estimated_obstacle_distances, dt, save_path='distances_plot.png'):
    time_steps = np.arange(len(goal_distances)) * dt

    obst_radius = 0.6

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(time_steps, goal_distances, label='Distance to Goal')

    # num_obstacles = estimated_obstacle_distances.shape[1]
    # for i in range(num_obstacles):
    #     plt.plot(time_steps, estimated_obstacle_distances[:, i] - obst_radius, label=f'Estimated Distance to Obstacle {i+1}')

    # plt.axhline(y=0.2, color='black', linestyle='--', linewidth=3, label='Safety Margin')

    plt.xlabel('Time (s)')
    plt.ylabel('Distance')
    plt.title('Distances to Goal and Estimated Distances to Obstacles over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


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

def sample_points_on_obstacle(obstacle, num_points_per_face):
    sampled_points = []
    for face in obstacle:
        # Calculate the min and max coordinates of the face
        min_coords = np.min(face, axis=0)
        max_coords = np.max(face, axis=0)

        # Generate random points within the face boundaries
        points = np.random.rand(num_points_per_face, 3)
        points[:, 0] = points[:, 0] * (max_coords[0] - min_coords[0]) + min_coords[0]
        points[:, 1] = points[:, 1] * (max_coords[1] - min_coords[1]) + min_coords[1]
        points[:, 2] = points[:, 2] * (max_coords[2] - min_coords[2]) + min_coords[2]

        sampled_points.append(points)

    return np.concatenate(sampled_points, axis=0)


def generate_realistic_env_3d(corridor_pos, corridor_size, obstacle_positions, obstacle_sizes, num_points_per_face=30):


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
    

    # Generate obstacle shapes based on positions and sizes
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

        sampled_points = sample_points_on_obstacle(obstacle, num_points_per_face=num_points_per_face)

        obstacle_points.append(sampled_points)


    return wall_positions, obstacle_shapes, np.array(obstacle_points).reshape(-1, 3)


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
    colors = ['red', 'green', 'blue', 'orange']
    
    for obstacle, color in zip(obstacle_shapes, colors):
        for face in obstacle:
            ax.add_collection3d(Poly3DCollection([face], facecolors=color, linewidths=1, edgecolors='k', alpha=0.6))

    # ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], c='black', s=5)

    # Plot goal position
    ax.scatter(*goal_position, c='blue', marker='*', s=100)

    plt.tight_layout()

    if plt_show:
        plt.show()

if __name__ == '__main__':
    # Define the environment parameters

    corridor_pos = np.array([0, 0, -1])

    corridor_size = np.array([2, 2, 2])

    obstacle_positions = [
        np.array([2.5, 0, 4.4]),
        np.array([2.5, 0, 0]),
        np.array([1, -4, 3]), 
        np.array([5.5, 0, 2])
    ]
    obstacle_sizes = [
        np.array([1, 5, 3]),
        np.array([1, 5, 2]),
        np.array([3, 2, 4]),
        np.array([2.5, 5, 1])
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
    
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-3, 9)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-2, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    cable_lengths = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.], [2.0, 2.0]])

    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH

    plot_links_3d(cable_lengths, link_radius, link_length, ax, base_center, base_normal)

    plot_env_3d(wall_positions, obstacle_shapes, goal_position, sphere_positions, sphere_radius, ax, plt_show = True)
