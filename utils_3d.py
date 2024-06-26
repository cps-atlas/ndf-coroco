import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, vmap
import jax

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

@jit
def calculate_link_parameters(edge_lengths, link_radius):
    q1, q2, q3 = edge_lengths
    r = link_radius
    # Calculate theta (bending angle)
    theta = 2 * jnp.sqrt(q1**2 + q2**2 + q3**2 - q1*q2 - q2*q3 - q1*q3) / (3 * r)
    # Calculate phi (bending direction angle)
    phi = jnp.arctan2(jnp.sqrt(3) * (q2 - q3), q2 + q3 - 2*q1)
    return theta, phi

@jit
def compute_edge_points(edge_lengths, link_radius, link_length):
    theta, phi = calculate_link_parameters(edge_lengths, link_radius)
    # Calculate the bending radius
    R = jnp.where(jnp.abs(theta) < 1e-4, jnp.inf, link_length / theta)
    # Generate points for the deformed link
    t = jnp.linspace(0, theta, 50)

    # Calculate points for each edge
    edge_points = []
    for i in range(3):
        # Calculate the angle for each cable
        cable_angle = i * 2 * jnp.pi / 3
        r = link_radius
        # Calculate the x, y, z coordinates for each point along the edge
        x = jnp.where(R == jnp.inf,
                      link_radius * jnp.cos(cable_angle) * jnp.ones_like(jnp.linspace(0, link_length, 50)),
                      R - (R - r * jnp.cos(cable_angle)) * jnp.cos(t))
        y = jnp.where(R == jnp.inf,
                      link_radius * jnp.sin(cable_angle) * jnp.ones_like(jnp.linspace(0, link_length, 50)),
                      r * jnp.sin(cable_angle) * jnp.ones_like(t))
        z = jnp.where(R == jnp.inf,
                      jnp.linspace(0, link_length, 50),
                      (R - r * jnp.cos(cable_angle)) * jnp.sin(t))
        # Rotate the points based on phi
        rotated_x = x * jnp.cos(phi) - y * jnp.sin(phi)
        rotated_y = x * jnp.sin(phi) + y * jnp.cos(phi)
        edge_points.append(jnp.vstack((rotated_x, rotated_y, z)))
    return edge_points



@jit
def compute_end_circle(states, link_radius, link_length, base_center=jnp.zeros(3), base_normal=jnp.array([0, 0, 1])):
    end_center = base_center
    end_normal = base_normal
    

    for state in states:
        edge_lengths = compute_3rd_edge_length(state, link_length)
        edge_points = jnp.array(compute_edge_points(edge_lengths, link_radius, link_length))  # Convert to jnp.ndarray

        rotation_matrix = jnp.where(
            jnp.array_equal(end_center, jnp.zeros(3)),
            jnp.eye(3),
            calculate_rotation_matrix(jnp.array([0, 0, 1]), end_normal)
        )
        
        # Use vmap to apply the transformations to each edge point
        edge_points = vmap(lambda points: jnp.dot(rotation_matrix, points))(edge_points)
        edge_points = vmap(lambda points: points + end_center.reshape(3, 1))(edge_points)

        end_center, end_normal, end_radius = calculate_link_circle(edge_points)

    return end_center, end_normal, end_radius


@jit
def calculate_link_circle(edge_points):
    # Get the three points that construct the end circle
    p1 = edge_points[0][:, -1]
    p2 = edge_points[1][:, -1]
    p3 = edge_points[2][:, -1]
    # Compute the center of the end circle
    end_center = jnp.mean(jnp.stack([p1, p2, p3]), axis=0)
    # Compute the normal vector of the end circle
    v1 = p2 - p1
    v2 = p3 - p1
    end_normal = jnp.cross(v1, v2)
    end_normal = end_normal / jnp.linalg.norm(end_normal)
    radius = jnp.linalg.norm(end_center - p1)

    return end_center, end_normal, radius

@jit
def calculate_rotation_matrix(v1, v2):
    v1 = v1 / jnp.linalg.norm(v1)
    v2 = v2 / jnp.linalg.norm(v2)
    v = jnp.cross(v1, v2)
    c = jnp.dot(v1, v2)
    s = jnp.linalg.norm(v)
    vx = jnp.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return jnp.where(s < 1e-6, jnp.eye(3), jnp.eye(3) + vx + jnp.dot(vx, vx) * (1 - c) / (s ** 2))


#@jit
def compute_3rd_edge_length(state, link_length):

    q1, q2 = state
    q3 = 3 * link_length - q1 - q2
    edge_lengths = [q1, q2, q3]

    return edge_lengths

    
def plot_circle(center, radius, normal, ax, color='k'):
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)

    # Rotate the circle points based on the normal vector
    rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), normal)
    rotated_points = np.dot(rotation_matrix, np.vstack((x, y, z)))

    # Translate the rotated points based on the center
    translated_points = rotated_points + center.reshape(3, 1)

    ax.plot(translated_points[0], translated_points[1], translated_points[2], color=color, linewidth=2)



def plot_links_3d(states, link_radius, link_length, ax, base_center=np.zeros(3), base_normal=np.array([0, 0, 1])):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    legend_elements = []

    for i, state in enumerate(states):
        edge_lengths = compute_3rd_edge_length(state, link_length)

        # Plot the base circle
        plot_circle(base_center, link_radius, base_normal, ax)

        # Compute the deformed link points for each edge
        edge_points = compute_edge_points(edge_lengths, link_radius, link_length)

        # Apply rotation and translation to the edge points
        if not np.array_equal(base_center, np.array([0, 0, 0])):
            rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), base_normal)
            for j in range(len(edge_points)):
                rotated_edge_points = np.dot(rotation_matrix, edge_points[j])
                translated_edge_points = rotated_edge_points + base_center.reshape(3, 1)
                edge_points[j] = translated_edge_points

        # Plot the deformed edges
        for points in edge_points:
            ax.plot(points[0], points[1], points[2], color=colors[i], linewidth=2)

        # Calculate the end circle center, normal, and radius
        end_center, end_normal, end_radius = calculate_link_circle(edge_points)

        # print('end_center:', end_center)
        # print('end_mormal:', end_normal)

        # Plot the end circle if it's the last link
        if i == len(states) - 1:
            plot_circle(end_center, end_radius, end_normal, ax)

        # Collect legend entry for the current link
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Link {i+1}'))

        # Update the base center and normal for the next link
        base_center = end_center
        base_normal = end_normal

    return legend_elements


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
        vel = np.array([0.0, 0.0, 0.0])

        obstacle_velocities.append(vel)

    obstacle_positions = np.array(obstacle_positions)

    obstacle_velocities = np.array(obstacle_velocities)

    # Generate random goal point
    while True:
        goal_point = np.random.uniform(low=[goal_xlim[0], goal_ylim[0], goal_zlim[0]], high=[goal_xlim[1], goal_ylim[1], goal_zlim[1]])
        if all(np.linalg.norm(goal_point - obs_pos) >= min_distance_goal for obs_pos in obstacle_positions):
            break

    return obstacle_positions, obstacle_velocities, goal_point


def main():
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Define the link parameters
    link_radius = 0.15
    link_length = 1.0

    # Define the states for multiple links
    states = np.array([[1.0, 1.0], [1.1, 1.0], [1.0, 1.0], [1.0, 1.0]])


    # Define the initial base center and normal
    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    end_center, end_normal, _ = compute_end_circle(states, link_radius, link_length)

    print(end_center)

    # Plot the links
    legend_elements = plot_links_3d(states, link_radius, link_length, ax, base_center, base_normal)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((4, 4, 4))

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig("3d_continuum_arm_multiple_links.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()