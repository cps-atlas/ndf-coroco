import numpy as np
import random
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_3d import *
from jax import jit

'''
if no GPU
'''
jax.config.update('jax_platform_name', 'cpu')

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

def prepare_training_data_3d(num_configs, length_range, num_links, link_radius, link_length, workspace_resolution, x_range, y_range, z_range):
    dataset = []

    # Generate grid points in the workspace
    workspace_points = generate_grid_points(x_range, y_range, z_range, workspace_resolution)

    for step in range(num_configs):
        print('config:', step)
        # Sample random edge lengths for each link
        states = np.zeros((num_links, 2))
        for i in range(num_links):
            l1 = np.random.uniform(length_range[0], length_range[1])
            l2_max = min(length_range[1], 3 * link_length - l1 - length_range[0])
            l2_min = max(length_range[0], 3 * link_length - l1 - length_range[1])
            l2 = np.random.uniform(l2_min, l2_max)
            states[i, :] = [l1, l2]

            # for debugging (just the upright config)
            # states[i, :] = [1., 1.]


        # Calculate theta and phi for each link
        thetas = []
        phis = []
        for state in states:
            edge_lengths = compute_3rd_edge_length(state, link_length)
            theta, phi = calculate_link_parameters(edge_lengths, link_radius)
            thetas.append(theta)
            phis.append(phi)

        # Generate surface points for each link
        surface_points_list = compute_surface_points(states, link_radius, link_length, num_points_per_circle=20)

        #print('surface_points_list:', surface_points_list)

        # Flatten the surface points list into a single array
        surface_points = np.concatenate(surface_points_list, axis=0)

        #print('surface_points:', surface_points)

        # print(thetas)
        # print(phis)

        # Flatten the configurations: order: theta1, phi1, theta2, phi2, ...
        configurations = np.stack((thetas, phis), axis=1).flatten()

        #print('confirgurations:', configurations)

        # Add surface points to the dataset with distance 0
        for point in surface_points:
            entry = {
                'configurations': configurations,
                'point': point,
                'distance': 0.
            }
            dataset.append(entry)

        # print('workspace_point:', workspace_points)
        # Compute distances for each workspace point
        for point in workspace_points:

            distance = compute_distance(point, surface_points_list)

            distance = np.asarray(distance)

            
            # print('distance:', distance)
            
        
            # Create a dataset entry
            entry = {
                'configurations': configurations,
                'point': point,
                'distance': distance
            }
            dataset.append(entry)

        #print('dataset:', dataset)
        # Save the dataset every 100 configurations
        if (step + 1) % 100 == 0 or step == num_configs - 1:
            with open(f'dataset_3d_large_{step // 100}.pickle', 'wb') as f:
                pickle.dump(dataset, f)
            dataset = []  # Reset the dataset for the next batch

    return dataset

def visualize_dataset_3d(dataset):
    # Select a random entry from the dataset
    entry = random.choice(dataset)
    configurations = entry['configurations']
    thetas = configurations[:len(configurations)//2]
    phis = configurations[len(configurations)//2:]
    point = entry['point']

    # Plot the 3D continuum arm configuration
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    #print(entry)


    # Plot the surface points
    surface_points = [entry['point'] for entry in dataset if entry['distance'] == 0]
    surface_points = np.array(surface_points)
    # print(surface_points.shape)
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], color='green', label='Surface Points', alpha=0.5, s=5)

    # Plot the workspace points with color based on distance
    workspace_points = [entry['point'] for entry in dataset if entry['distance'] != 0]
    workspace_points = np.array(workspace_points)
    distances = [np.min(entry['distance']) for entry in dataset if entry['distance'] != 0]
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
    ax.set_title('Dataset Visualization')
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='green', linestyle='', alpha=0.5, markersize=5, label='Surface Points'),
                                         plt.Line2D([0], [0], marker='o', color='red', linestyle='', alpha=0.5, markersize=5, label='Workspace Points')], loc='upper right')

    plt.tight_layout()

    plt.savefig("3d_dataset.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # Example usage
    num_configs = 1000
    length_range = [0.8, 1.2]
    num_links = 4
    link_radius = 0.15
    link_length = 1.0
    workspace_resolution = 36
    x_range = [-4, 4]
    y_range = [-4, 4]
    z_range = [-1.5, 4.5]

    prepare_training_data_3d(num_configs, length_range, num_links, link_radius, link_length, workspace_resolution, x_range, y_range, z_range)

    # Concatenate the dataset files
    dataset = []
    for i in range(num_configs // 100 + 1):
        file_name = f'dataset_3d_large_{i}.pickle'
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                dataset.extend(pickle.load(f))

            os.remove(file_name)  # Remove the partial dataset file

    # Save the concatenated dataset to a file
    with open('dataset_3d_large.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    # Visualize the dataset
    # visualize_dataset_3d(dataset)