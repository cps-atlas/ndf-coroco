import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import jit, lax
import jax.numpy as jnp

from network.csdf_net import CSDFNet, CSDFNet_JAX

from training.config import *

@jit
def evaluate_model(jax_params, rbt_config, obstacle_points):
    # Predict signed distances
    @jit
    def apply_model(params, inputs):
        return CSDFNet_JAX(HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS).apply(params, inputs)

    # Ensure obstacle_points is a 2D array
    obstacle_points = jnp.array(obstacle_points, dtype=jnp.float32)
    if obstacle_points.ndim == 1:
        obstacle_points = obstacle_points.reshape(1, -1)

    # Prepare the input tensor
    configurations = jnp.repeat(jnp.array(rbt_config)[None, :], len(obstacle_points), axis=0)
    points = obstacle_points
    inputs = jnp.concatenate((configurations, points), axis=1)

    # Forward pass
    outputs = apply_model(jax_params, inputs)


    # Compute the gradients using JAX automatic differentiation
    gradients = []
    for i in range(len(obstacle_points)):
        point_gradients = []
        for j in range(outputs.shape[1]):  # Loop over the output dimension
            grad_fn = jax.grad(lambda x: apply_model(jax_params, x)[j])
            point_gradients.append(grad_fn(inputs[i]))
        gradients.append(jnp.array(point_gradients))
    gradients = jnp.array(gradients)


    # Extract the distances, gradients w.r.t robot configuration, and gradients w.r.t obstacle points
    link_gradients = gradients[:, :, :len(rbt_config)]
    obst_gradients = gradients[:, :, len(rbt_config):]

    

    return outputs, link_gradients, obst_gradients

@jit
def compute_cbf_value_and_grad(jax_params, rbt_config, obstacle_points, obstacle_velocities):
    # Compute the distances and gradients from each link to each obstacle point
    link_distances, link_gradients, obst_gradients = evaluate_model(jax_params, rbt_config, obstacle_points)



    # Ensure link_distances has the correct shape (num_obstacle_points, num_links)
    assert link_distances.shape[0] == len(obstacle_points), "Mismatch in number of obstacle points"
    assert link_distances.shape[1] == len(rbt_config), "Mismatch in number of links"

    # Find the minimum distance for each obstacle across all links
    min_indices = jnp.argmin(link_distances, axis=1)

    # Extract the minimum distances and corresponding gradients
    cbf_h_vals = link_distances[jnp.arange(len(obstacle_points)), min_indices]
    cbf_h_grads = link_gradients[jnp.arange(len(obstacle_points)), min_indices]

    # print('cbf_h_vals:', cbf_h_vals)
    # print('cbf_h_grads:', cbf_h_grads)

    # Compute the CBF time derivative (cbf_t_grad) for each obstacle
    cbf_t_grads = jnp.array([obst_gradients[i, min_indices[i]] @ obstacle_velocities[i] for i in range(len(obstacle_points))])


    return cbf_h_vals, cbf_h_grads, cbf_t_grads


@jit
def calculate_link(left_base, right_base, nominal_length, left_length):
    # Calculate link parameters
    link_width = jnp.sqrt((right_base[0] - left_base[0])**2 + (right_base[1] - left_base[1])**2)
    base_center = (left_base + right_base) / 2

    # Calculate the right edge length and curvature
    right_length = 2 * nominal_length - left_length

    # Calculate the slope and angle of the line connecting the left and right bases
    base_angle = jnp.arctan2(right_base[1] - left_base[1], right_base[0] - left_base[0])

    num_of_edge_pts = 80

    # Check if the link is deformed
    is_deformed = jnp.abs(right_length - left_length) >= 1e-6

    # Generate points for the original link
    theta_range = jnp.linspace(0, nominal_length, num_of_edge_pts)
    x_left_original = left_base[0] + theta_range * jnp.cos(base_angle + jnp.pi/2)
    y_left_original = left_base[1] + theta_range * jnp.sin(base_angle + jnp.pi/2)
    x_right_original = right_base[0] + theta_range * jnp.cos(base_angle + jnp.pi/2)
    y_right_original = right_base[1] + theta_range * jnp.sin(base_angle + jnp.pi/2)

    left_end_original = jnp.array([x_left_original[-1], y_left_original[-1]])
    right_end_original = jnp.array([x_right_original[-1], y_right_original[-1]])

    # Calculate the radius and center of the deformed link
    radius = (link_width / 2) * jnp.abs((left_length + right_length) / (right_length - left_length))



    center_x = base_center[0] - jnp.sign(right_length - left_length) * radius * jnp.cos(base_angle)
    center_y = base_center[1] - jnp.sign(right_length - left_length) * radius * jnp.sin(base_angle)

    # Calculate the central angle (theta) based on the arc length and radius
    theta = nominal_length / radius

    # Calculate the starting angle of the arc
    start_angle = jnp.arctan2(base_center[1] - center_y, base_center[0] - center_x)

    # Generate points for the deformed link
    theta_range_deformed = jnp.where(left_length < right_length,
                                     jnp.linspace(start_angle, start_angle + theta, num_of_edge_pts),
                                     jnp.linspace(start_angle, start_angle - theta, num_of_edge_pts))

    x_left_deformed = jnp.where(left_length < right_length,
                                center_x + (radius - link_width / 2) * jnp.cos(theta_range_deformed),
                                center_x + (radius + link_width / 2) * jnp.cos(theta_range_deformed))
    y_left_deformed = jnp.where(left_length < right_length,
                                center_y + (radius - link_width / 2) * jnp.sin(theta_range_deformed),
                                center_y + (radius + link_width / 2) * jnp.sin(theta_range_deformed))
    x_right_deformed = jnp.where(left_length < right_length,
                                 center_x + (radius + link_width / 2) * jnp.cos(theta_range_deformed),
                                 center_x + (radius - link_width / 2) * jnp.cos(theta_range_deformed))
    y_right_deformed = jnp.where(left_length < right_length,
                                 center_y + (radius + link_width / 2) * jnp.sin(theta_range_deformed),
                                 center_y + (radius - link_width / 2) * jnp.sin(theta_range_deformed))

    left_end_deformed = jnp.array([x_left_deformed[-1], y_left_deformed[-1]])
    right_end_deformed = jnp.array([x_right_deformed[-1], y_right_deformed[-1]])

    # Select the appropriate points based on the deformation condition
    left_end = jnp.where(is_deformed, left_end_deformed, left_end_original)
    right_end = jnp.where(is_deformed, right_end_deformed, right_end_original)
    x_left = jnp.where(is_deformed, x_left_deformed, x_left_original)
    y_left = jnp.where(is_deformed, y_left_deformed, y_left_original)
    x_right = jnp.where(is_deformed, x_right_deformed, x_right_original)
    y_right = jnp.where(is_deformed, y_right_deformed, y_right_original)

    # Return the left edge points from top to bottom, right edge points from bottom to top
    return left_end, right_end, jnp.vstack((x_left[::-1], y_left[::-1])).T, jnp.vstack((x_right, y_right)).T
    


@jit
def get_last_link_grasp_points(left_base, right_base, link_lengths, nominal_length):
    link_lengths_jax = jnp.array(link_lengths)

    def body_fun(i, inputs):
        left_base, right_base = inputs
        left_length = lax.dynamic_slice(link_lengths_jax, (i,), (1,))[0]
        left_end, right_end, _, _ = calculate_link(left_base, right_base, nominal_length, left_length)
        return left_end, right_end

    left_base, right_base = lax.fori_loop(0, len(link_lengths) - 1, body_fun, (left_base, right_base))

    left_length = link_lengths[-1]
    left_end, right_end, left_coords, right_coords = calculate_link(left_base, right_base, nominal_length, left_length)

    # Calculate the index for the point at 1/3 of the link's length
    right_index = len(right_coords) // 6
    left_index = len(left_coords) - right_index - 1

    left_point = left_coords[left_index]
    right_point = right_coords[right_index]

    return left_point, right_point


def plot_links(link_lengths, nominal_length, left_base, right_base, ax):
    # Define a list of colors for each link
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    legend_elements = []

    robot_left_base = left_base
    robot_right_base = right_base

    # Iterate over the link lengths and plot each link
    for i, left_length in enumerate(link_lengths):
        # Calculate the current link
        left_end, right_end, left_coords, right_coords = calculate_link(left_base, right_base, nominal_length, left_length)

        # Concatenate the left and right edge coordinates
        x_coords = np.concatenate((left_coords[:, 0], right_coords[:, 0]))
        y_coords = np.concatenate((left_coords[:, 1], right_coords[:, 1]))

        # Plot the deformed link
        fill = ax.fill(x_coords, y_coords, colors[i % len(colors)], alpha=0.4, edgecolor=colors[i % len(colors)], linewidth=3, label=f'Link {i+1}')
        
        # Collect legend entry for the current link
        legend_elements.append(fill[0])

        # Update the base points for the next link
        left_base = left_end
        right_base = right_end

    # Get the left and right middle points of the last link
    left_middle, right_middle = get_last_link_grasp_points(robot_left_base, robot_right_base, link_lengths, nominal_length)



    # Plot the left middle point
    # ax.plot(left_middle[0], left_middle[1], 'ko', markersize=8, label='Left Grasp')

    # # Plot the right middle point
    # ax.plot(right_middle[0], right_middle[1], 'ko', markersize=8, label='Right Grasp')

    ax.set_xlim(-4, 4)
    #ax.set_ylim(-len(link_lengths) * nominal_length + 0.5, len(link_lengths) * nominal_length + 0.5)
    ax.set_ylim(-2, len(link_lengths) * nominal_length + 0.5)
    # Increase font size for axis labels
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    
    # Increase font size for axis numbers
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Increase font size for legend
    ax.legend(fontsize=18)

    ax.set_aspect('equal')

    return legend_elements, left_coords, right_coords


def generate_random_env(num_obstacles, xlim_left, xlim_right, ylim, goal_xlim_left, goal_xlim_right, goal_ylim, min_distance_obs, min_distance_goal):
    # Generate random obstacle positions and velocities
    obstacle_positions = []
    obstacle_velocities = []

    for _ in range(num_obstacles):
        while True:
            # Randomly choose left or right side for obstacle position
            if np.random.rand() < 0.5:
                pos = np.random.uniform(low=[xlim_left[0], ylim[0]], high=[xlim_left[1], ylim[1]])
            else:
                pos = np.random.uniform(low=[xlim_right[0], ylim[0]], high=[xlim_right[1], ylim[1]])

            if all(np.linalg.norm(pos - np.array(obs_pos)) >= min_distance_obs for obs_pos in obstacle_positions):
                obstacle_positions.append(pos)
                break



        vel = np.random.uniform(low=[-0.5, -0.5], high=[0.5, 0.5])

        # static obstacle
        # vel = np.array([0.0, 0.0])

        obstacle_velocities.append(vel)

    obstacle_positions = np.array(obstacle_positions)

    obstacle_velocities = np.array(obstacle_velocities)



    # Generate random goal point
    while True:
        # Randomly choose left or right side for goal position
        if np.random.rand() < 0.5:
            goal_point = np.random.uniform(low=[goal_xlim_left[0], goal_ylim[0]], high=[goal_xlim_left[1], goal_ylim[1]])
        else:
            goal_point = np.random.uniform(low=[goal_xlim_right[0], goal_ylim[0]], high=[goal_xlim_right[1], goal_ylim[1]])

        if all(np.linalg.norm(goal_point - obs_pos) >= min_distance_goal for obs_pos in obstacle_positions):
            break

    # Hard-code obstacle positions and velocities
    obstacle_positions = [
        np.array([1.5, 3.5]),  # First obstacle
        np.array([1.3, -0.8])   # Second obstacle
    ]
    obstacle_velocities = [
        np.array([0.5, 0.0]),  # First obstacle moves horizontally
        np.array([0.0, 0.5])   # Second obstacle moves vertically
    ]

    obstacle_positions = np.array(obstacle_positions)
    obstacle_velocities = np.array(obstacle_velocities)

    goal_point = np.array([3.2, 0.8])

    return obstacle_positions, obstacle_velocities, goal_point



def main():
    # Define the sequence of link lengths
    link_lengths = [0.8, 0.9, 1.1, 1]

    #link_lengths = [1.0, 1.0, 1.0, 1.35]
    nominal_length = 1.0

    # Define the initial base points for the first link
    left_base = [-0.15, 0.0]
    right_base = [0.15, 0.0]

    left_base = jnp.array(left_base)
    right_base = jnp.array(right_base)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the links
    plot_links(link_lengths, nominal_length, left_base, right_base, ax)

    plt.show()

if __name__ == '__main__':
    main()