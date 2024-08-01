import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_utils import save_high_res_figure, sample_points_on_face, generate_sphere_points
from utils_3d import *
from robot_config import *




def generate_charging_port_env_3d(corridor_pos, corridor_size, charging_port_position, charging_port_size, obst_points_per_unit):
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
        # Right face
        # np.array([
        #     [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        #     [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        #     [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        #     [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        # ]),
    ]

    pos = charging_port_position

    size = charging_port_size

    charging_port_shape = [
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
        # Right face
        np.array([
            [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] - size[2] / 2],
            [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] - size[2] / 2],
            [pos[0] + size[0] / 2, pos[1] + size[1] / 2, pos[2] + size[2] / 2],
            [pos[0] + size[0] / 2, pos[1] - size[1] / 2, pos[2] + size[2] / 2],
        ]),
    ]

    left_wall_positions = []

    # Top rectangle
    left_wall_positions.append(np.array([
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, charging_port_position[2] + charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, charging_port_position[2] + charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
    ]))

    # Bottom rectangle
    left_wall_positions.append(np.array([
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, charging_port_position[2] - charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, charging_port_position[2] - charging_port_size[2] / 2],
    ]))

    # Left rectangle
    left_wall_positions.append(np.array([
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, charging_port_position[2] - charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, charging_port_position[1] - charging_port_size[1] / 2, charging_port_position[2] - charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, charging_port_position[1] - charging_port_size[1] / 2, charging_port_position[2] + charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, charging_port_position[2] + charging_port_size[2] / 2],
    ]))

    # Right rectangle
    left_wall_positions.append(np.array([
        [corridor_pos[0] - corridor_size[0] / 2, charging_port_position[1] + charging_port_size[1] / 2, charging_port_position[2] - charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, charging_port_position[2] - charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, charging_port_position[2] + charging_port_size[2] / 2],
        [corridor_pos[0] - corridor_size[0] / 2, charging_port_position[1] + charging_port_size[1] / 2, charging_port_position[2] + charging_port_size[2] / 2],
    ]))

    wall_positions.extend(left_wall_positions)

    # floor_positions = []

    # # Top rectangle
    # floor_positions.append(np.array([
    #     [-1, -4, 0],
    #     [-1, 4, 0],
    #     [7, 4, 0],
    #     [7, -4, 0],
    # ]))

    # wall_positions.extend(floor_positions)

    obstacle_points = []
    sampled_points = []
    for face in wall_positions:
        face_points = sample_points_on_face(face, obst_points_per_unit)
            
        sampled_points.append(face_points)

    # Concatenate the sampled points for the current obstacle
    obstacle_points.append(np.concatenate(sampled_points, axis=0))

    # Concatenate all obstacle points into a single array
    obstacle_points = np.concatenate(obstacle_points, axis=0)

    # append right face

    wall_positions_right = [
        np.array([
        [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] - corridor_size[2] / 2],
        [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] + corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
        [corridor_pos[0] + corridor_size[0] / 2, corridor_pos[1] - corridor_size[1] / 2, corridor_pos[2] + corridor_size[2] / 2],
    ])]

    wall_positions.extend(wall_positions_right)


    return wall_positions, charging_port_shape, obstacle_points




def plot_charging_env(wall_positions, charging_port_shape, sphere_positions, sphere_radius, ax, plt_show=False):
    for wall in wall_positions:
        ax.add_collection3d(Poly3DCollection([wall], facecolors='gray', linewidths=1, edgecolors='k', alpha=0.3))

    for face in charging_port_shape:
        ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, edgecolors='k', alpha=0.8))

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

    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    if plt_show:
        plt.show()

def plot_and_save_charging_port_env():
    wall_position = np.array([5, 0, 3])
    wall_size = np.array([1.0, 8, 6])

    charging_port_size = np.array([1.0, 1., 1.])
    # Define the fixed x position for the charging port
    x_position = wall_position[0]
    
    # Generate random y and z positions within the specified ranges
    y_position = np.random.uniform(-3, 3)
    z_position = np.random.uniform(1, 5)
    
    # Construct the charging port position
    charging_port_position = np.array([x_position, y_position, z_position])

    wall_positions, charging_port_shape, obstacle_points = generate_charging_port_env_3d(wall_position, wall_size, charging_port_position, charging_port_size, obst_points_per_unit=5)



    # Generate dynamic spheres
    sphere_positions = np.array([
        np.array([3.0, -4.0, 4.5]),  # Sphere 1 position
        np.array([3.0, -4.0, 3.5]),   # Sphere 2 position
        np.array([3.0, -4.0, 2.5]),   # Sphere 3 position
        np.array([3.0, -4.0, 1.5]),   # Sphere 3 position
        np.array([3.0, -4.0, 0.5])   # Sphere 3 position
    ])
    sphere_velocities = np.array([
        np.array([0.0, 1.0, 0.0]),  # Sphere 1 velocity
        np.array([0.0, 1.0, 0.0]),  # Sphere 2 velocity
        np.array([0.0, 1.0, 0.0]),  # Sphere 1 velocity
        np.array([0.0, 1.0, 0.0]),  # Sphere 2 velocity
        np.array([0.0, 1.0, 0.0])  # Sphere 2 velocity
    ])

    
    sphere_radius = 0.5

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

    cable_lengths = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH

    plot_links_3d(cable_lengths, link_radius, link_length, ax, base_center, base_normal)

    # Plot obstacle points correctly
    ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], c='r')

    plot_charging_env(wall_positions, charging_port_shape, sphere_positions, sphere_radius, ax, plt_show = True)



    save_high_res_figure(fig, '3d_charging_port_environment.png')
    plt.close(fig)


if __name__ == '__main__':


    # plot an env with charging port
    plot_and_save_charging_port_env()