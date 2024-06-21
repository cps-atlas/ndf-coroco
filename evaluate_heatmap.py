import torch

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.config import *

from utils import plot_links

from flax.core import freeze

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

def evaluate_csdf(net, configuration, x_range, y_range, resolution=100, model_type='jax'):
    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx.flatten(), yy.flatten()), axis=1)

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

    distances = min_sdf_distance.reshape(resolution, resolution)
    return distances


def plot_csdf_heatmap(configuration, distances, x_range, y_range, nominal_length, left_base, right_base, save_path=None):
    # Plot the soft robot configuration
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_links(configuration, nominal_length, left_base, right_base, ax)

    # Plot the signed distance field heatmap
    im = ax.imshow(distances, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], cmap='coolwarm', alpha=0.7)
    fig.colorbar(im, ax=ax, label='Signed Distance')

    # Plot the zero level set
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], distances.shape[0]),
                         np.linspace(y_range[0], y_range[1], distances.shape[1]))
    contour = ax.contour(xx, yy, distances, levels=[0], colors='k', linewidths=2)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
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
        net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)
        net.load_state_dict(torch.load("trained_models/torch_models/trained_with_normal.pth"))
        net.eval()
    elif model_type == 'jax':
        # Load the trained JAX model
        net = CSDFNet_JAX(HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)
        net.params = jnp.load("trained_models/jax_models/trained_no_eikonal.npy", allow_pickle=True).item()


        pytorch_net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)

        # Load state_dict from file
        state_dict = torch.load("trained_models/torch_models/baseline_2.pth")
        pytorch_net.load_state_dict(state_dict)

        # Define your JAX model
        jax_net = CSDFNet_JAX(HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)

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
    y_range = (-1, 5)

    # Define the nominal length and initial base points
    nominal_length = 1.0
    left_base = np.array([-0.15, 0.0])
    right_base = np.array([0.15, 0.0])

    # Define the configuration to evaluate
    configuration = np.array([1.1, 1.1, 1.1, 1.1])

    # Evaluate the signed distance values
    distances = evaluate_csdf(net, configuration, x_range, y_range, resolution=200, model_type=model_type)

    # Plot the C-SDF heatmap
    plot_csdf_heatmap(configuration, distances, x_range, y_range, nominal_length, left_base, right_base, save_path='csdf_heatmap.png')

if __name__ == "__main__":
    main()