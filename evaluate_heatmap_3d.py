import torch

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.config_3D import *

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils_3d import *

from flax.core import freeze

from robot_config import *

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

def evaluate_csdf_3d(net, configurations, cable_lengths, points, link_radius, link_length, resolution=None, model_type='jax'):
    num_links = len(configurations)
    num_points = points.shape[0]

    # Compute the transformations using forward kinematics
    transformations = forward_kinematics(cable_lengths)
    # Exclude the end-effector transformation
    transformations = transformations[:-1]

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

def plot_csdf_heatmap_3d(cable_lengths, distances, link_radius, link_length, x_range, y_range, z_range, save_path=None):
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    plot_links_3d(cable_lengths, link_radius, link_length, ax, base_center, base_normal)

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

    # Clip the distances to the threshold value
    clipped_distances = np.clip(distances, -threshold, threshold)

    # Create a colormap that emphasizes color changes near the threshold
    cmap = plt.get_cmap('coolwarm')
    normalized_distances = (clipped_distances + threshold) / (2 * threshold)

    # Plot the heatmap of the signed distance values
    sc = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=distances.flatten(), cmap=cmap, alpha=0.2, s=4)
    cbar = fig.colorbar(sc, ax=ax, label='Distance')


    # Highlight points with distance < 0.3
    condition = distances < threshold
    sc_highlight = ax.scatter(xx[condition], yy[condition], zz[condition], c='blue', marker='o', s=20, alpha=0.3, label='Distance < 0.3')

    # Create a colorbar with adjusted limits
    #cbar.set_ticks([-1, 0, 1])
    #cbar.set_ticklabels([f'<= -{threshold}', '0', f'>= {threshold}'])


    # Set the limits and labels of the plot
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title('C-SDF Heatmap')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the figure in high resolution

    plt.show()


def surface_point_csdf_check(configuration, cable_lengths, link_radius, link_length, net, model_type):

    surface_points_list = compute_surface_points([cable_lengths[0]], link_radius, link_length, num_points_per_circle=30)
    surface_points = np.concatenate(surface_points_list, axis=0)

    # Evaluate the distances for the surface points
    surface_distances = evaluate_csdf_3d(net, configuration, cable_lengths, surface_points, link_radius, link_length, model_type=model_type)

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




def main():
    # Define the model type to evaluate ('torch' or 'jax')
    model_type = 'jax'

    trained_model = "trained_models/torch_models_3d/eikonal_train_32.pth"

    net = load_learned_csdf(model_type, trained_model_path = trained_model)

    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH

    # Define the cable lengths for each link
    cable_lengths = jnp.array([[2.0, 2.0], [1.8, 1.9], [1.9, 2.1], [2.2, 2.0]])

    #cable_lengths = jnp.array([[2.2, 2.0], [2.1, 2.0]])

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


    surface_point_csdf_check(configuration, cable_lengths, link_radius, link_length, net, model_type)


    points_of_interest = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [2, 0, 0],
        [2, 2, 0], 
        [0, 0, 4]
    ])

    # Evaluate the distances for the points of interest
    distances = evaluate_csdf_3d(net, configuration, cable_lengths, points_of_interest, link_radius, link_length, model_type=model_type)

    # Print the distances for each point of interest
    for point, distance in zip(points_of_interest, distances):
        print(f"Point: {point}, Distance: {distance:.4f}")


    '''
    plot the heatmap for visualization 
    '''

    # Define the region of interest
    x_range = (-4, 4)
    y_range = (-4, 4)
    z_range = (-2, 6)

    resolution = 40

    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)

    distances = evaluate_csdf_3d(net, configuration, cable_lengths, points, link_radius, link_length, model_type=model_type, resolution=resolution)

    # Plot the C-SDF isosurface
    plot_csdf_heatmap_3d(cable_lengths, distances, link_radius, link_length, x_range, y_range, z_range, save_path='csdf_isosurface_3d.png')


if __name__ == "__main__":
    main()