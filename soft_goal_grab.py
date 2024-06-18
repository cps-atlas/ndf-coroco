import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch

import jax
import jax.numpy as jnp
import flax
import flax.traverse_util as traverse_util

from visualize_soft_link import plot_links
import imageio

from utils.csdf_net import CSDFNet, CSDFNet_JAX
from training.config import *

from main_csdf import evaluate_model, compute_cbf_value_and_grad


from control.clf_qp import ClfQpController
from control.clf_cbf_qp import ClfCbfController

from mppi_functional import setup_mppi_controller

from flax.core import freeze


def main(jax_params, goal_point, dt):
    num_links = 4
    num_steps = 200
    nominal_length = 1.0

    left_base = [-0.15, 0.0]
    right_base = [0.15, 0.0]

    link_lengths = np.ones(num_links) * nominal_length

    writer = imageio.get_writer("grab_animation.mp4", fps=int(1/dt))
    fig, ax = plt.subplots(figsize=(8, 8))

    clf_qp_controller = ClfQpController()

    for _ in range(num_steps):
        # Clear the previous plot
        ax.clear()

        sdf_val, sdf_grad, _ = evaluate_model(jax_params, link_lengths, goal_point)

        # Plot the links using the plot_links function
        legend_elements = plot_links(link_lengths, nominal_length, left_base, right_base, ax)

        # Plot the goal point
        goal_plot, = ax.plot(goal_point[0], goal_point[1], marker='o', markersize=15, color='blue', label='Goal')
        legend_elements.append(goal_plot)

        # Convert the plot to an image and append it to the video
        ax.legend(handles=legend_elements, fontsize=16)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        writer.append_data(image[:, :, :3])


        if sdf_val[-1][-1] < 0.3:
            # Implement grasping strategy
            link_lengths[-1] *= 1.01  # Bend the last link
        else:
            # Use clf_qp controller to reach the goal

            control_signals = clf_qp_controller.generate_controller(link_lengths, sdf_val[-1][-1], sdf_grad[-1][-1])
            link_lengths += control_signals * dt


        # if reached the goal
        if sdf_val[-1][-1] < 0.1:
            print("Goal Reached!")
            # Freeze the video for an additional 0.5 second
            for _ in range(int(0.5 / dt)):
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                writer.append_data(image[:, :, :3])
            break

    writer.close()

if __name__ == '__main__':
    goal_point = np.array([2.8, 1.5])
    dt = 0.05

    pytorch_net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)

    # Load state_dict from file
    state_dict = torch.load("trained_models/torch_models/trained_model_eikonal.pth")
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

    # simulate the control performance
    main(jax_params, goal_point, dt)