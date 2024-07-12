import numpy as np
import matplotlib.pyplot as plt

from network.csdf_net import CSDFNet_JAX
from training.config_3D import *
from robot_config import *

import jax

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import jit, vmap, lax

@jit
def calculate_link_parameters(edge_lengths, link_radius):
    q1 = edge_lengths[..., 0]
    q2 = edge_lengths[..., 1]
    q3 = edge_lengths[..., 2]
    r = link_radius
    
    # Calculate theta (bending angle)
    theta_expr = q1**2 + q2**2 + q3**2 - q1*q2 - q2*q3 - q1*q3
    theta_expr = jnp.maximum(theta_expr, 0.0)  # Ensure non-negative value inside sqrt
    theta = 2 * jnp.sqrt(theta_expr) / (3 * r)
    
    # Calculate phi (bending direction angle)
    phi = jnp.arctan2(jnp.sqrt(3) * (q2 - q3), q2 + q3 - 2*q1)

    '''
    rrt* paper definition:
    '''
    # theta = 2 * jnp.sqrt(q1**2 + q2**2 - q1 * q2) / (jnp.sqrt(3) * r)

    # phi = jnp.arctan2(jnp.sqrt(3) * (q2), 2*q1 - q2)
    # jax.debug.print("🤯 theta: {theta}, phi: {phi}", theta = theta, phi = phi)
    
    return theta, phi


@jit
def compute_edge_points(edge_lengths, link_radius, link_length):

    num_pts_per_edge = 30

    theta, phi = calculate_link_parameters(edge_lengths, link_radius)
    # Calculate the bending radius
    R = jnp.where(jnp.abs(theta) < 1e-10, jnp.inf, link_length / theta)
    # Generate points for the deformed link
    t = jnp.linspace(0, theta, num_pts_per_edge)

    # Calculate points for each edge
    edge_points = []
    for i in range(3):
        # Calculate the angle for each cable
        cable_angle = i * 2 * jnp.pi / 3
        r = link_radius
        # Calculate the x, y, z coordinates for each point along the edge
        x = jnp.where(R == jnp.inf,
                      link_radius * jnp.cos(cable_angle) * jnp.ones_like(jnp.linspace(0, link_length, num_pts_per_edge)),
                      R - (R - r * jnp.cos(cable_angle)) * jnp.cos(t))
        y = jnp.where(R == jnp.inf,
                      link_radius * jnp.sin(cable_angle) * jnp.ones_like(jnp.linspace(0, link_length, num_pts_per_edge)),
                      r * jnp.sin(cable_angle) * jnp.ones_like(t))
        z = jnp.where(R == jnp.inf,
                      jnp.linspace(0, link_length, num_pts_per_edge),
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



@jit
def compute_3rd_edge_length(state, link_length):
    q1 = state[..., 0]
    q2 = state[..., 1]
    q3 = 3 * link_length - q1 - q2
    edge_lengths = jnp.stack([q1, q2, q3], axis=-1)
    return edge_lengths


@jit
def state_to_config(edge_lengths, link_radius, link_length):
    edge_lengths_3 = compute_3rd_edge_length(edge_lengths, link_length)
    thetas, phis = calculate_link_parameters(edge_lengths_3, link_radius)
    return jnp.stack((thetas, phis), axis=-1)


'''
for following functions, NUM_OF_LINKS < 10 is relatively small, explicit for loops may be the most efficient implementation
'''

@jit
def forward_kinematics(states, link_radius, link_length, base_center = jnp.zeros(3), base_normal = jnp.array([0, 0, 1])):

    transformations = [jnp.eye(4)]

    for i in range(len(states)):

        # Compute the end circle and normal for the current link
        end_center, end_normal, _ = compute_end_circle(states[:i+1], link_radius, link_length, base_center, base_normal)

        rotation_matrix = calculate_rotation_matrix(base_normal, end_normal)
        translation = end_center - base_center

        transformation = jnp.eye(4)
        transformation = transformation.at[:3, :3].set(rotation_matrix)
        transformation = transformation.at[:3, 3].set(translation)

        transformations.append(transformation)

    return transformations

@jit
def transform_point_to_link_frame(point, transformations):
    homogeneous_point = jnp.append(point, 1)
    link_frames = []

    for transformation in transformations:   
        link_frame_point = jnp.dot(jnp.linalg.inv(transformation), homogeneous_point)
        link_frames.append(link_frame_point[:3])

    return jnp.array(link_frames)



'''
following function is the inference of learned C-SDF
'''

@jit
def evaluate_model(jax_params, rbt_configs, cable_lengths, link_radius, link_length, obstacle_points):
    # Predict signed distances
    @jit
    def apply_model(params, inputs):
        return CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).apply(params, inputs)

    # Ensure obstacle_points is a 2D array
    obstacle_points = jnp.array(obstacle_points, dtype=jnp.float32)
    if obstacle_points.ndim == 1:
        obstacle_points = obstacle_points.reshape(1, -1)

    rbt_configs = rbt_configs.reshape(NUM_OF_LINKS, 2)
    num_links = len(rbt_configs)
    num_points = obstacle_points.shape[0]

    # Compute the transformations using forward kinematics
    transformations = forward_kinematics(cable_lengths, link_radius, link_length)
    # Exclude the end-effector transformation
    transformations = transformations[:-1]

    # Convert transformations to an ndarray
    transformations = jnp.array(transformations)

    # Transform the points to all link frames simultaneously
    points_link = jax.vmap(lambda T: jnp.dot(jnp.linalg.inv(T), jnp.hstack((obstacle_points, jnp.ones((num_points, 1)))).T).T[:, :3])(transformations)

    # Prepare the input tensor for all links
    inputs_link = jnp.concatenate((jnp.repeat(rbt_configs, num_points, axis=0), jnp.reshape(points_link, (-1, 3))), axis=1)

    # Forward pass for all links
    outputs_link = apply_model(jax_params, inputs_link)
    distances_link = outputs_link[:, 0].reshape(num_links, num_points)

    # Find the minimum distances and closest link indices
    min_distances = jnp.min(distances_link, axis=0)


    return min_distances

'''
following are non JAX functions, mainly for plotting and dataset preparation
'''


def compute_surface_points(states, link_radius, link_length, num_points_per_circle=50):
    surface_points = []
    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    # Sample points on the base circle of the first link
    base_disk_points = sample_disk_points(base_center, link_radius, base_normal, num_points=100)
    surface_points.append(base_disk_points)


    for state in states:
        edge_lengths = compute_3rd_edge_length(state, link_length)
        edge_points = compute_edge_points(edge_lengths, link_radius, link_length)

        # Apply rotation and translation to the edge points
        if not np.array_equal(base_center, np.array([0, 0, 0])):
            rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), base_normal)
            for j in range(len(edge_points)):
                rotated_edge_points = np.dot(rotation_matrix, edge_points[j])
                translated_edge_points = rotated_edge_points + base_center.reshape(3, 1)
                edge_points[j] = translated_edge_points

        link_surface_points = []

        for i in range(len(edge_points[0][0])):
            # Get the three edge points at the current index
            p1 = np.array([edge_points[0][0][i], edge_points[0][1][i], edge_points[0][2][i]])
            p2 = np.array([edge_points[1][0][i], edge_points[1][1][i], edge_points[1][2][i]])
            p3 = np.array([edge_points[2][0][i], edge_points[2][1][i], edge_points[2][2][i]])

            # Compute the center, normal, and radius of the circle
            center = np.mean([p1, p2, p3], axis=0)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            radius = np.linalg.norm(center - p1)

            # Generate points on the circle
            theta = np.linspace(0, 2*np.pi, num_points_per_circle)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.zeros_like(theta)

            # Rotate the circle points based on the normal vector
            rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), normal)
            rotated_points = np.dot(rotation_matrix, np.vstack((x, y, z)))

            # Translate the rotated points based on the center
            translated_points = rotated_points + center.reshape(3, 1)

            # Append the circle points to the link surface points
            link_surface_points.extend(translated_points.T)

        surface_points.append(np.array(link_surface_points))

        # Update the base center and normal for the next link
        end_center, end_normal, _ = calculate_link_circle(edge_points)
        base_center = end_center
        base_normal = end_normal

    # Sample points on the end circle of the last link
    end_disk_points = sample_disk_points(end_center, link_radius, end_normal, num_points=100)
    surface_points.append(end_disk_points)

    return surface_points

def sample_disk_points(center, radius, normal, num_points=100):
    # Sample random radii within the disk
    radii = np.sqrt(np.random.uniform(0, radius**2, size=num_points))
    
    # Sample random angles
    angles = np.random.uniform(0, 2*np.pi, size=num_points)
    
    # Convert radii and angles to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.zeros_like(radii)
    
    # Stack the coordinates
    disk_points = np.vstack((x, y, z))
    
    # Rotate the disk points based on the normal vector
    rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), normal)
    rotated_points = np.dot(rotation_matrix, disk_points)
    
    # Translate the rotated points based on the center
    translated_points = rotated_points + center.reshape(3, 1)
    
    return translated_points.T

    
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





def main():
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Define the link parameters
    link_radius = 0.15
    link_length = 1.0

    # Define the states for multiple links
    states = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    # debug state
    # states =  jnp.array([[1.005272,  1.0100657] ,[1.      ,  0.999946] , [0.9991085, 0.9851592], [0.9849205,
    # 0.99     ]])


    # Define the initial base center and normal
    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    end_center, end_normal, _ = compute_end_circle(states, link_radius, link_length)

    transformations = forward_kinematics(states, link_radius, link_length)

    print('transmations:', transformations)

    point_ws = np.array([2,2,0])

    link_frames = transform_point_to_link_frame(point_ws, transformations)

    print('link_frames:', link_frames)


    # Plot the links
    legend_elements = plot_links_3d(states, link_radius, link_length, ax, base_center, base_normal)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-2, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((4, 4, 4))

    # ax.view_init(elev=20, azim=45)  # Adjust the elevation and azimuth angles as needed

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig("3d_continuum_arm_multiple_links.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()