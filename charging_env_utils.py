import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from plot_utils import save_high_res_figure
from utils_3d import *
from robot_config import *




def generate_charging_port_env_3d(corridor_pos, corridor_size, charging_port_position, charging_port_size):
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

    return wall_positions, charging_port_shape




def plot_charging_env(wall_positions, charging_port_shape, ax, plt_show=False):
    for wall in wall_positions:
        ax.add_collection3d(Poly3DCollection([wall], facecolors='gray', linewidths=1, edgecolors='k', alpha=0.8))

    for face in charging_port_shape:
        ax.add_collection3d(Poly3DCollection([face], facecolors='green', linewidths=1, edgecolors='k', alpha=0.8))

    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    if plt_show:
        plt.show()

def plot_and_save_charging_port_env():
    wall_position = np.array([5, 0, 3])
    wall_size = np.array([1.0, 10, 6])
    charging_port_position = np.array([5, 0, 3])
    charging_port_size = np.array([1.0, 1, 1])

    wall_positions, charging_port_shape = generate_charging_port_env_3d(wall_position, wall_size, charging_port_position, charging_port_size)

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

    plot_charging_env(wall_positions, charging_port_shape, ax, plt_show = True)

    save_high_res_figure(fig, '3d_charging_port_environment.png')
    plt.close(fig)


if __name__ == '__main__':


    # plot an env with charging port
    plot_and_save_charging_port_env()