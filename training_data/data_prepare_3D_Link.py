import numpy as np
import random
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_3d import *
from robot_config import *
from jax import jit

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp

def generate_grid_points(x_range, y_range, z_range, resolution):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
    return points

@jit
def compute_distance(point, surface_points):
    distances = jnp.array([jnp.min(jnp.linalg.norm(point - surface_point, axis=1)) for surface_point in surface_points])
    distance = jnp.min(distances)
    return distance

def prepare_training_data_3d_single_link(num_configs, length_range, link_radius, link_length, workspace_resolution, x_range, y_range, z_range):
    dataset = []

    # Generate grid points in the workspace
    workspace_points = generate_grid_points(x_range, y_range, z_range, workspace_resolution)

    for step in range(num_configs):
        print('config:', step)
        # Sample random edge lengths for the link
        l1 = np.random.uniform(length_range[0], length_range[1])
        l2_max = min(length_range[1], 3 * link_length - l1 - length_range[0])
        l2_min = max(length_range[0], 3 * link_length - l1 - length_range[1])
        l2 = np.random.uniform(l2_min, l2_max)
        state = np.array([l1, l2])

        # Calculate theta and phi for the link
        edge_lengths = compute_3rd_edge_length(state, link_length)
        theta, phi = calculate_link_parameters(edge_lengths, link_radius)

        # Generate surface points for the link
        surface_points_list = compute_surface_points([state], link_radius, link_length, num_points_per_circle=40)

        # Flatten the surface points list into a single array
        surface_points = np.concatenate(surface_points_list, axis=0)

        # Configuration: theta, phi
        configuration = np.array([theta, phi])

        # Add surface points to the dataset with distance 0
        for point in surface_points:
            entry = {
                'configuration': configuration,
                'point': point,
                'distance': 0.
            }
            dataset.append(entry)

        # Compute distances for each workspace point
        for point in workspace_points:
            distance = compute_distance(point, surface_points_list)
            distance = np.asarray(distance)

            # Create a dataset entry
            entry = {
                'configuration': configuration,
                'point': point,
                'distance': distance
            }
            dataset.append(entry)

    return dataset

def visualize_dataset_3d_single_link(dataset):
    # Select a random entry from the dataset
    entry = random.choice(dataset)
    configuration = entry['configuration']
    theta, phi = configuration

    # Filter the dataset to include only entries with the selected configuration
    filtered_dataset = [entry for entry in dataset if np.array_equal(entry['configuration'], configuration)]

    # Plot the 3D continuum arm configuration
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface points
    surface_points = [entry['point'] for entry in filtered_dataset if entry['distance'] == 0]
    surface_points = np.array(surface_points)
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], color='green', label='Surface Points', alpha=0.5, s=5)

    # Plot the workspace points with color based on distance
    workspace_points = [entry['point'] for entry in filtered_dataset if entry['distance'] != 0]
    workspace_points = np.array(workspace_points)
    distances = [np.min(entry['distance']) for entry in filtered_dataset if entry['distance'] != 0]
    distances = np.array(distances)
    normalized_distances = distances / np.max(distances)
    colors = plt.cm.jet(normalized_distances)
    ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], c=colors, label='Workspace Points', alpha=0.5, s=5)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Dataset Visualization (theta={theta:.2f}, phi={phi:.2f})')
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='green', linestyle='', alpha=0.5, markersize=5, label='Surface Points'),
                        plt.Line2D([0], [0], marker='o', color='red', linestyle='', alpha=0.5, markersize=5, label='Workspace Points')], loc='upper right')

    plt.tight_layout()

    plt.savefig("3d_dataset_single_link.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # Example usage
    num_configs = 250
    length_range = [MIN_CABLE_LENGTH, MAX_CABLE_LENGTH]
    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH
    workspace_resolution = 32
    x_range = [-LINK_LENGTH - 0.5, LINK_LENGTH + 0.5]
    y_range = [-LINK_LENGTH - 0.5, LINK_LENGTH + 0.5]
    z_range = [-1, LINK_LENGTH + 1]

    dataset = prepare_training_data_3d_single_link(num_configs, length_range, link_radius, link_length, workspace_resolution, x_range, y_range, z_range)


    visualize_dataset_3d_single_link(dataset)
    # Save the entire dataset to a single file
    with open('dataset_3d_single_link_large.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    