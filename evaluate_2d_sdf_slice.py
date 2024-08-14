import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from network.csdf_net import CSDFNet_JAX
from training.config_3D import *
from utils_3d import *
from flax.core import freeze
from robot_config import *

from evaluate_heatmap import load_learned_csdf

def evaluate_csdf_2d(net, configurations, cable_lengths, points, resolution, model_type='jax'):
    num_links = len(configurations)
    num_points = points.shape[0]

    # Compute the transformations using forward kinematics
    transformations = forward_kinematics(cable_lengths)

    # Initialize the minimum distance array
    min_distances = jnp.full(num_points, jnp.inf)

    for i in range(num_links):
        # Transform the points to the current link's local frame
        points_link = jnp.dot(jnp.linalg.inv(transformations[i]), jnp.hstack((points, jnp.zeros((num_points, 1)))).T).T[:, :3]

        # Prepare the input tensor for the current link
        inputs_link = jnp.hstack((jnp.repeat(configurations[i].reshape(1, -1), num_points, axis=0), points_link))

        # Evaluate the signed distance values for the current link
        if model_type == 'jax':
            outputs_link = net.apply(net.params, inputs_link)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Supported type is 'jax'.")

        # Update the minimum distance array
        min_distances = jnp.minimum(min_distances, jnp.min(outputs_link, axis=1))

    distances = min_distances.reshape(resolution, resolution)
    return distances
def plot_csdf_2d(distances, x_range, y_range, slice_index, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    im = ax.imshow(distances, cmap='viridis', origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'C-SDF Slice {slice_index}')

    fig.colorbar(im, ax=ax, label='Distance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def main():
    model_type = 'jax'
    trained_model = "trained_models/torch_models_3d/grid_search_4_16.pth"

    net = load_learned_csdf(model_type, trained_model_path=trained_model)


    cable_lengths = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    thetas = []
    phis = []
    for length in cable_lengths:
        edge_lengths = compute_3rd_edge_length(length)
        theta, phi = calculate_link_parameters(edge_lengths)
        thetas.append(theta)
        phis.append(phi)

    configuration = np.stack((thetas, phis), axis=1)

    z_range = (-1, 9)
    x_slices = [-1, 0, 1]
    y_range = (-5, 5)

    resolution = 100

    for slice_index, x_slice in enumerate(x_slices):
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        yy, zz = np.meshgrid(y, z, indexing='ij')
        points = np.stack((np.full(yy.flatten().shape, x_slice), yy.flatten(), zz.flatten()), axis=1)

        distances = evaluate_csdf_2d(net, configuration, cable_lengths, points, resolution, model_type=model_type)

        plot_csdf_2d(distances, y_range, z_range, slice_index, save_path=f'csdf_slice_x_{slice_index}.png')

if __name__ == "__main__":
    main()