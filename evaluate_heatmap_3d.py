import torch

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.config_3D import *

from skimage import measure

from utils_3d import *

from flax.core import freeze

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

def evaluate_csdf_3d(net, configuration, points, resolution = None, model_type='jax'):

    # Prepare the input tensor
    configurations = np.repeat(configuration.reshape(1, -1), points.shape[0], axis=0)

    inputs = np.hstack((configurations, points))

    if model_type == 'torch':
        inputs_tensor = torch.from_numpy(inputs).float()
        # Evaluate the signed distance values
        net.eval()
        with torch.no_grad():
            outputs = net(inputs_tensor)
        min_sdf_distance = np.min(outputs.numpy(), axis=1)
    elif model_type == 'jax':
        inputs_tensor = jnp.array(inputs)
        # Evaluate the signed distance values
        outputs = net.apply(net.params, inputs)
        min_sdf_distance = np.min(np.array(outputs), axis=1)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Supported types are 'torch' and 'jax'.")

    if resolution is None:
        return min_sdf_distance
    else:
        distances = min_sdf_distance.reshape(resolution, resolution, resolution)
        return distances

def plot_csdf_heatmap_3d(configuration, distances, x_range, y_range, z_range, save_path=None):
    # Create a figure and 3D axis
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    link_radius = 0.15
    link_length = 1.0

    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    plot_links_3d(configuration, link_radius, link_length, ax, base_center, base_normal)

    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], distances.shape[0])
    y = np.linspace(y_range[0], y_range[1], distances.shape[1])
    z = np.linspace(z_range[0], z_range[1], distances.shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Plot the heatmap of the signed distance values
    cmap = plt.get_cmap('coolwarm')
    sc = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=distances.flatten(), cmap=cmap, alpha=0.1, s=1)
    fig.colorbar(sc, ax=ax, label='Distance')

    # # Set the limits and labels of the plot
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('C-SDF Heatmap')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the figure in high resolution
    plt.show()

def main():
    # Define the model type to evaluate ('torch' or 'jax')
    model_type = 'jax'

    if model_type == 'torch':
        # Load the trained PyTorch model
        net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        net.load_state_dict(torch.load("trained_models/torch_models_3d/new_test.pth"))
        net.eval()
    elif model_type == 'jax':
        # Load the trained JAX model
        net = CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        # net.params = jnp.load("trained_models/jax_models_3d/new_test.npy", allow_pickle=True).item()


        pytorch_net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

        # Load state_dict from file
        state_dict = torch.load("trained_models/torch_models_3d/test_1.pth")
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

    # Define the region of interest
    x_range = (-4, 4)
    y_range = (-4, 4)
    z_range = (-1, 4)

    resolution = 50

    # Define the cable lengths for each link
    link_lengths = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    # Calculate theta and phi for each link
    thetas = []
    phis = []
    link_radius = 0.15
    link_length = 1.0
    for length in link_lengths:
        edge_lengths = compute_3rd_edge_length(length, link_length)
        theta, phi = calculate_link_parameters(edge_lengths, link_radius)
        thetas.append(theta)
        phis.append(phi)

    # Flatten the configurations
    configuration = np.stack((thetas, phis), axis=1).flatten()

    points_of_interest = np.array([
        [0, 0, 0],
        [0, 0, -1],
        [2, 0, 0],
        [2, 2, 0], 
        [0, 0, 4]
    ])

    # Evaluate the distances for the points of interest
    distances = evaluate_csdf_3d(net, configuration, points_of_interest, model_type=model_type)

    # Print the distances for each point of interest
    for point, distance in zip(points_of_interest, distances):
        print(f"Point: {point}, Distance: {distance:.4f}")


    '''
    plot the heatmap for visualization 
    '''

    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)

    distances = evaluate_csdf_3d(net, configuration, points, model_type=model_type, resolution=50)

    

    # Plot the C-SDF isosurface
    plot_csdf_heatmap_3d(link_lengths, distances, x_range, y_range, z_range, save_path='csdf_isosurface_3d.png')


if __name__ == "__main__":
    main()