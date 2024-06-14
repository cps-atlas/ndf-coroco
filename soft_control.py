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

def integrate_link_lengths(link_lengths, control_signals, dt):
    # Euler integration to update link lengths
    link_lengths += control_signals * dt
    return link_lengths

def main(jax_params, obstacle_position, obstacle_velocity, goal_point, dt, control_mode = 'clf_cbf'):
    # Define the number of links and time steps
    num_links = 4
    num_steps = 100 #20
    
    nominal_length = 1.0

    obst_radius = 0.3

    # Define the initial base points for the first link
    left_base = [-0.15, 0.0]
    right_base = [0.15, 0.0]

    # Initialize link lengths
    link_lengths = np.ones(num_links) * nominal_length


    # Create controllers for each control mode
    clf_cbf_controller = ClfCbfController()
    clf_qp_controller = ClfQpController()

    # Simulate and record videos for each control mode
    mode = control_mode
    
    link_lengths = np.ones(num_links) * nominal_length

    # Create a video writer object for each control mode
    writer = imageio.get_writer(f"{mode}_animation.mp4", fps=int(1/dt))
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial control input guess for MPPI

    prediction_horizon = 10 #20

    U = 0.04 * jnp.ones((prediction_horizon, 4))

    for _ in range(num_steps):

        # Compute the signed distance and gradient to the goal point
        sdf_val, sdf_grad, _ = evaluate_model(jax_params, link_lengths, goal_point)


        if sdf_val[-1][-1] < 0.1:
            print("Goal Reached!")
            # Freeze the video for an additional 0.5 second
            for _ in range(int(0.5 / dt)):
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                writer.append_data(image[:, :, :3])
            break

        if mode == 'clf_cbf':
            # Compute the CBF value and gradients
            cbf_h_val, cbf_h_grad, cbf_t_grad = compute_cbf_value_and_grad(jax_params, link_lengths, obstacle_position, obstacle_velocity)

            # Generate control signals using the CBF-CLF controller
            control_signals = clf_cbf_controller.generate_controller(link_lengths, cbf_h_val - obst_radius, cbf_h_grad, cbf_t_grad, sdf_val[-1][-1], sdf_grad[-1][-1])

            # Update obstacle position
            
            obstacle_position += obstacle_velocity * dt

        elif mode == 'clf_qp':  # mode == 'clf_qp'


            # Generate control signals using the CLF-QP controller
            control_signals = clf_qp_controller.generate_controller(link_lengths, sdf_val[-1][-1], sdf_grad[-1][-1])

        elif mode == 'mppi':   # mode = MPPI
            num_samples = 2000 #20000
            costs_lambda = 0.03
            cost_goal_coeff = 15.0
            cost_safety_coeff = 1.0
            cost_perturbation_coeff = 0.1
            cost_goal_coeff_final = 15.0
            cost_safety_coeff_final = 1.0
            cost_state_coeff = 10.0



            mppi = setup_mppi_controller(learned_CSDF = jax_params, horizon=prediction_horizon, samples=num_samples, input_size=4, control_bound=0.2, dt=dt, u_guess=None, use_GPU=False, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final, cost_state_coeff=cost_state_coeff)
            key = jax.random.PRNGKey(111)


            key, subkey = jax.random.split(key)

            robot_sampled_states, robot_selected_states, control_signals, U = mppi(subkey, U, link_lengths, goal_point, obstacle_position)

            control_signals = control_signals.reshape(4)
            print('controls:', control_signals)

            # Update obstacle position
            obstacle_position += obstacle_velocity * dt

        # Update link lengths using Euler integration

       
        link_lengths = integrate_link_lengths(link_lengths, control_signals, dt)


        # if robot stuck, just break
        if max(abs(control_signals)) < 5e-3:
            print("Robot Stucked!")
            # Freeze the video for an additional 0.5 second
            for _ in range(int(0.5 / dt)):
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                writer.append_data(image[:, :, :3])
            break

        

        # Clear the previous plot
        ax.clear()

        # Plot the links using the plot_links function
        legend_elements = plot_links(link_lengths, nominal_length, left_base, right_base, ax)

        if mode == 'clf_cbf' or 'mppi':
            
            goal_plot, = ax.plot(goal_point[0], goal_point[1], marker='*', markersize=15, color='blue', label = 'Goal')
            legend_elements.append(goal_plot)
            # Plot the obstacles
            obstacle_plots = []
            for i, obstacle in enumerate(obstacle_position):
                obstacle_circle = Circle((obstacle[0], obstacle[1]), radius=obst_radius, color='r', fill=True)
                ax.add_patch(obstacle_circle)
                if i == 0:
                    obstacle_plot = ax.plot([obstacle[0]], [obstacle[1]], 'ro', markersize=10, label='Obstacles')[0]
                    legend_elements.append(obstacle_plot)

            if obstacle_plots:
                ax.legend(handles=legend_elements)
        
        elif mode == 'clf_qp':  # mode == 'clf_qp'
            # Plot the goal point
            ax.plot(goal_point[0], goal_point[1], marker='*', markersize=15, color='blue', label = 'Goal')

        # Convert the plot to an image and append it to the video
        ax.legend(handles=legend_elements, fontsize=16)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        writer.append_data(image[:, :, :3])

    writer.close()

if __name__ == '__main__':
    # Define obstacle initial position and velocity
    # obstacle_positions = np.array([[2.4, 3.2]])
    # obstacle_velocities = np.array([[0.0, 0.0]])

    obstacle_positions = np.array([
        [2.0, 3.1],
        #[-1.0, 2.5],
        [1.0, 0.2]
    ])
    obstacle_velocities = np.array([
        [0.0, 0.0],
        #[0.0, 0.0],
        [0.0, 0.0]
    ])

    # Define the goal point
    goal_point = np.array([2.8, 1.5])

    # time step 
    dt = 0.05 

    control_mode = 'mppi'
    # control_mode = 'clf_cbf'


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
    main(jax_params, obstacle_positions, obstacle_velocities, goal_point, dt,control_mode)