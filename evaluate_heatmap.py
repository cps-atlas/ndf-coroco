import torch

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.config_3D import *

import pickle
from utils_3d import *

from flax.core import freeze

from robot_config import *

import time

from training_data.dataset import SoftRobotDataset

from torch.utils.data import DataLoader

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

def load_learned_csdf(model_type, trained_model_path = "trained_models/torch_models_3d/eikonal_train.pth"):

    if model_type == 'torch':
        # Load the trained PyTorch model
        net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        net.load_state_dict(torch.load(trained_model_path))
        net.eval()
    elif model_type == 'jax':
        # Load the trained JAX model
        net = CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        # net.params = jnp.load("trained_models/jax_models_3d/new_test.npy", allow_pickle=True).item()


        pytorch_net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

        # Load state_dict from file
        state_dict = torch.load(trained_model_path)
        pytorch_net.load_state_dict(state_dict)

        # Define your JAX model
        jax_net = CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

        # Initialize JAX model parameters
        rng = jax.random.PRNGKey(0)
        jax_params = jax_net.init(rng, jnp.zeros((1, INPUT_SIZE)))


        # Transfer weights from PyTorch to JAX
        def transfer_weights(pytorch_dict):
            new_params = {'params': {}}
            for i in range(NUM_LAYERS):
                new_params['params'][f'Dense_{i}'] = {
                    'kernel': jnp.array(pytorch_dict[f'hidden_layers.{i}.weight'].T),
                    'bias': jnp.array(pytorch_dict[f'hidden_layers.{i}.bias'])
                }
            return freeze(new_params)
        # Initialize JAX model parameters
        rng = jax.random.PRNGKey(0)
        _, jax_params = jax_net.init_with_output(rng, jnp.zeros((1, INPUT_SIZE)))

        # Transfer weights from PyTorch to JAX
        jax_params = transfer_weights(pytorch_net.state_dict())

        net.params = jax_params


    else:
        raise ValueError(f"Invalid model type: {model_type}. Supported types are 'torch' and 'jax'.")
    
    return net

def evaluate_csdf_3d(net, configurations, cable_lengths, points, resolution=None, model_type='jax'):
    num_links = len(configurations)
    num_points = points.shape[0]

    # Compute the transformations using forward kinematics
    transformations = forward_kinematics(cable_lengths)
    # Exclude the end-effector transformation
    # transformations = transformations[:-1]

    # Initialize the minimum distance array
    min_distances = jnp.full(num_points, jnp.inf)


    for i in range(num_links):
        # Transform the points to the current link's local frame
        points_link = jnp.dot(jnp.linalg.inv(transformations[i]), jnp.hstack((points, jnp.ones((num_points, 1)))).T).T[:, :3]


        # Prepare the input tensor for the current link
        inputs_link = jnp.hstack((jnp.repeat(configurations[i].reshape(1, -1), num_points, axis=0), points_link))
        
        # Evaluate the signed distance values for the current link
        if model_type == 'jax':
            outputs_link = net.apply(net.params, inputs_link)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Supported type is 'jax'.")

        # Update the minimum distance array
        min_distances = jnp.minimum(min_distances, jnp.min(outputs_link, axis=1))


    if resolution is None:
        return min_distances
    else:
        distances = min_distances.reshape(resolution, resolution, resolution)
        return distances

def plot_csdf_heatmap_3d(distances, x_range, y_range, z_range, save_path=None):
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')


    # plot_links_3d(cable_lengths, link_radius, link_length, ax, base_center, base_normal)

    # Plot surface points as green dots
    # surface_points_list = compute_surface_points([cable_lengths[0]], link_radius, link_length, num_points_per_circle=20)
    # surface_points = np.concatenate(surface_points_list, axis=0)
    # ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], c='green', marker='o', s=10, label='Surface Points')


    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], distances.shape[0])
    y = np.linspace(y_range[0], y_range[1], distances.shape[1])
    z = np.linspace(z_range[0], z_range[1], distances.shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Set the distance threshold for emphasizing color changes
    threshold = 0.2

    # Define distance ranges and corresponding colors
    distance_ranges = [
        (0, 0.4, 'blue', 0.75),
        (0.4, 0.8, 'orange', 0.5),
        (0.8, 1.2, 'red', 0.3),
    ]

    for min_dist, max_dist, color, alpha in distance_ranges:
        condition = (distances >= min_dist) & (distances < max_dist)
        sc = ax.scatter(xx[condition], yy[condition], zz[condition], 
                        c=color, marker='o', s=20, alpha=alpha, 
                        label=f'{min_dist:.1f} ≤ d < {max_dist:.1f}')
        
    ax.set_axis_off()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the figure in high resolution

    plt.show()


def surface_point_csdf_check(configuration, cable_lengths, link_radius, link_length, net, model_type):

    surface_points_list = compute_surface_points([cable_lengths[0]], link_radius, link_length, num_points_per_circle=30)
    surface_points = np.concatenate(surface_points_list, axis=0)

    # Evaluate the distances for the surface points
    surface_distances = evaluate_csdf_3d(net, configuration, cable_lengths, surface_points, model_type=model_type)

    # Compute the mean and standard deviation of the surface distances
    mean_surface_distance = np.mean(surface_distances)
    std_surface_distance = np.std(surface_distances)

    print(f"Mean surface distance: {mean_surface_distance:.4f}")
    print(f"Standard deviation of surface distances: {std_surface_distance:.4f}")

    # Check if the surface distances are close to 0
    epsilon = 1e-2  # Adjust this threshold as needed
    if np.abs(mean_surface_distance) < epsilon and std_surface_distance < epsilon:
        print("Surface distances are close to 0. Training is accurate.")
    else:
        print("Surface distances are not close to 0. Training may need improvement.")

def compute_safety_margin(net, val_dataloader):
    predicted_distances = []
    true_distances = []
    for val_inputs, val_targets in val_dataloader:
        val_inputs = val_inputs.detach().numpy()  # Detach and convert to numpy array
        val_inputs = jnp.array(val_inputs)
        val_outputs = net.apply(net.params, val_inputs).squeeze()
        # Store the predicted and true distances
        predicted_distances.extend(val_outputs.tolist())
        true_distances.extend(val_targets.detach().numpy())  # Detach and convert to numpy array

    # Convert distances to numpy arrays
    predicted_distances = np.array(predicted_distances)
    true_distances = np.array(true_distances)

    # Compute the safety margin for each validation sample
    safety_margins = np.maximum(0, true_distances - predicted_distances)

    # Compute the average safety margin
    avg_safety_margin = np.mean(safety_margins)

    return avg_safety_margin


def measure_inference_time(net, configurations, cable_lengths, points, model_type='jax', num_iterations=100):
    num_links = len(configurations)
    num_points = points.shape[0]

    # Compute the transformations using forward kinematics
    transformations = forward_kinematics(cable_lengths)

    start_time = time.time()
    for _ in range(num_iterations):
        for i in range(num_links):
            # Transform the points to the current link's local frame
            points_link = jnp.dot(jnp.linalg.inv(transformations[i]), jnp.hstack((points, jnp.ones((num_points, 1)))).T).T[:, :3]

            # Prepare the input tensor for the current link
            inputs_link = jnp.hstack((jnp.repeat(configurations[i].reshape(1, -1), num_points, axis=0), points_link))

            # Evaluate the signed distance values for the current link
            if model_type == 'jax':
                _ = net.apply(net.params, inputs_link)
            else:
                raise ValueError(f"Invalid model type: {model_type}. Supported type is 'jax'.")

    end_time = time.time()
    average_inference_time = (end_time - start_time) / num_iterations
    print(f"Average inference time for {num_points} points over {num_iterations} iterations: {average_inference_time:.4f} seconds")




def main():
    # Define the model type to evaluate ('torch' or 'jax')
    model_type = 'jax'

    # trained_model = "trained_models/torch_models_3d/eikonal_train_4_16.pth"

    # paper table prepare
    trained_model = "trained_models/torch_models_3d/grid_search_4_16.pth"

    net = load_learned_csdf(model_type, trained_model_path = trained_model)

    # load the validation dataset
    with open('training_data/dataset_3d_single_link_validation1.pickle', 'rb') as f:
        validation_data = pickle.load(f)


    # Extract configurations, workspace points, and distances from the dataset
    configurations = np.array([entry['configuration'] for entry in validation_data])
    points = np.array([entry['point'] for entry in validation_data])
    distances = np.array([entry['distance'] for entry in validation_data])

    val_dataset = SoftRobotDataset(configurations, points, distances)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    safety_margin = compute_safety_margin(net, val_dataloader)

    # Print the safety margin
    print(f"Safety Margin: {safety_margin:.4f}")

    # Define the cable lengths for each link
    cable_lengths = jnp.array([[1.8, 2.0], [1.9, 1.9], [2.1, 2.1], [2.2, 2.0]])

    # link_radius = LINK_RADIUS
    # link_length = LINK_LENGTH

    # fig = plt.figure(figsize=(8, 8), dpi=150)
    # ax = fig.add_subplot(111, projection='3d')

    # plot_links_3d(cable_lengths, link_radius, link_length, ax)

    # # Remove axis
    # ax.set_axis_off()

    # plt.show()

    # Calculate theta and phi for each link
    thetas = []
    phis = []
    for length in cable_lengths:
        edge_lengths = compute_3rd_edge_length(length)
        theta, phi = calculate_link_parameters(edge_lengths)
        thetas.append(theta)
        phis.append(phi)

    # Flatten the configurations
    configuration = np.stack((thetas, phis), axis=1)


    # Measure inference time for a single link
    num_points = 1000  # Adjust the number of points as needed
    points = np.random.rand(num_points, 3)
    measure_inference_time(net, configuration, cable_lengths, points, model_type=model_type, num_iterations=100)



    points_of_interest = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [2, 0, 0],
        [2, 2, 0], 
        [0, 0, 4]
    ])

    # Evaluate the distances for the points of interest
    distances = evaluate_csdf_3d(net, configuration, cable_lengths, points_of_interest, model_type=model_type)

    # Print the distances for each point of interest
    for point, distance in zip(points_of_interest, distances):
        print(f"Point: {point}, Distance: {distance:.4f}")


    '''
    plot the heatmap for visualization 
    '''

    # Define the region of interest
    x_range = (-6, 6)
    y_range = (-6, 6)
    z_range = (-2, 8)

    resolution = 40

    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)

    distances = evaluate_csdf_3d(net, configuration, cable_lengths, points, model_type=model_type, resolution=resolution)

    # Plot the C-SDF isosurface
    plot_csdf_heatmap_3d(distances, x_range, y_range, z_range, save_path='csdf_isosurface_3d.png')


if __name__ == "__main__":
    main()