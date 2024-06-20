import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch

import jax
import jax.numpy as jnp

from utils import *
import imageio

from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.config import *

from main_csdf import evaluate_model, compute_cbf_value_and_grad


from control.clf_qp import ClfQpController
from control.clf_cbf_qp import ClfCbfController

from control.mppi_functional import setup_mppi_controller

from flax.core import freeze


def main(jax_params, obstacle_position, obstacle_velocity, goal_point, dt, control_mode='clf_cbf', env_idx=0, trial_idx=0):

    # initialize the paarameters for return
    success_count = 0
    collision_count = 0
    total_time = 0.0


    # Define the number of links and time steps
    num_links = 4
    num_steps = 200
    
    nominal_length = 1.0

    obst_radius = 0.3

    # Define the initial base points for the first link
    left_base = jnp.array([-0.15, 0.0])
    right_base = jnp.array([0.15, 0.0])

    # Initialize link lengths
    link_lengths = np.ones(num_links) * nominal_length

    # different initial states
    # link_lengths = np.array([1.05, 0.95, 0.85, 0.95])




    # Create controllers for each control mode
    clf_cbf_controller = ClfCbfController()
    clf_qp_controller = ClfQpController()

    # Simulate and record videos for each control mode
    mode = control_mode

    # Create directory structure for saving videos
    result_dir = 'result_videos'
    env_dir = f'env{env_idx+1}'
    mode_dir = control_mode
    video_name = f'trial{trial_idx+1}.mp4'

    os.makedirs(os.path.join(result_dir, env_dir, mode_dir), exist_ok=True)
    video_path = os.path.join(result_dir, env_dir, mode_dir, video_name)

    # Create a video writer object for each control mode
    writer = imageio.get_writer(video_path, fps=int(1/dt))
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial control input guess for MPPI

    prediction_horizon = 30 #20

    U = 0.01 * jnp.ones((prediction_horizon, num_links))

    grab_mode = False

    grab_successful = False

    grab_goal = np.copy(goal_point)
    grab_goal[1] += 0.01

    if grab_goal[0] < 0:
        grab_goal[0] -= 0.35
    else:
        grab_goal[0] += 0.35

    # Determine the grabbing control signal based on the grab point
    grab_control = -0.15 if grab_goal[0] < 0 else 0.15

    # Define the initial control signal
    control_signals = np.random.uniform(-0.12, 0.12, size=num_links)

    for step in range(num_steps):

        # draw the video
        # Clear the previous plot
        ax.clear()

        # Plot the links using the plot_links function
        legend_elements = plot_links(link_lengths, nominal_length, left_base, right_base, ax)

        
        if mode in ['clf_cbf', 'mppi']:    
            goal_plot, = ax.plot(goal_point[0], goal_point[1], marker='*', markersize=12, color='blue', label = 'Goal')
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
            ax.plot(goal_point[0], goal_point[1], marker='*', markersize=12, color='blue', label = 'Goal')

        # Convert the plot to an image and append it to the video
        ax.legend(handles=legend_elements, fontsize=16)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        writer.append_data(image[:, :, :3])

        # Freeze the video at the initial states for 0.5 seconds
        if step == 0:
            for _ in range(int(0.5 / dt)):
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                writer.append_data(image[:, :, :3])

        # compute the last link's middle points 
        grab_left_point, grab_right_point  = get_last_link_middle_points(left_base, right_base, link_lengths, nominal_length)

        #grab_point = grab_left_point if np.linalg.norm(grab_left_point - grab_goal) < np.linalg.norm(grab_right_point - grab_goal) else grab_right_point

        grab_point = grab_right_point 

        grab_distance = np.linalg.norm(grab_point - grab_goal)
        print('grab_distance:', grab_distance)




        # Compute the signed distance and gradient to the goal point
        sdf_val, sdf_grad, _ = evaluate_model(jax_params, link_lengths, goal_point)



        # Check for collision
        if mode == 'clf_cbf' or mode == 'mppi':
            cbf_h_val, cbf_h_grad, cbf_t_grad = compute_cbf_value_and_grad(jax_params, link_lengths, obstacle_position, obstacle_velocity)

            # print('cbf_h_val:', cbf_h_val)
            
            if min(cbf_h_val) - obst_radius < 0:
                collision_count += 1
                print("Collision detected!")
                break

        if mode == 'clf_cbf':
            # Compute the CBF value and gradients
            cbf_h_val, cbf_h_grad, cbf_t_grad = compute_cbf_value_and_grad(jax_params, link_lengths, obstacle_position, obstacle_velocity)

            # Generate control signals using the CBF-CLF controller
            control_signals = clf_cbf_controller.generate_controller(link_lengths, cbf_h_val - obst_radius, cbf_h_grad, cbf_t_grad, sdf_val[-1][-1], sdf_grad[-1][-1])

            # Update obstacle position
            
            obstacle_position += obstacle_velocity * dt

            if sdf_val[-1][-1] < 0.1:
                print("Goal Reached!")
                # Freeze the video for an additional 0.5 second
                success_count = 1
                for _ in range(int(0.5 / dt)):
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    writer.append_data(image[:, :, :3])
                break

        elif mode == 'clf_qp':  # mode == 'clf_qp'


            # Generate control signals using the CLF-QP controller
            control_signals = clf_qp_controller.generate_controller(link_lengths, sdf_val[-1][-1], sdf_grad[-1][-1])


        elif mode == 'random':  # mode == random

            # Generate a random perturbation vector
            perturbation = np.random.uniform(-0.04, 0.04, size=num_links)
    
            # Update the control signal with the perturbation
            control_signals = control_signals + perturbation
    
            # Clip the control signal to the desired range
            control_signals = np.clip(control_signals, -0.12, 0.12)


        elif mode == 'mppi':   # mode = MPPI
            num_samples = 5000      #20000
            costs_lambda = 0.03
            cost_goal_coeff = 18.0
            cost_safety_coeff = 2.2
            cost_goal_coeff_final = 15.0
            cost_safety_coeff_final = 1.8
            cost_state_coeff = 10.0



            

            if not grab_mode and grab_distance >= 0.1:
                mppi = setup_mppi_controller(learned_CSDF=jax_params, horizon=prediction_horizon, samples=num_samples, input_size=4, control_bound=0.2, dt=dt, u_guess=None, use_GPU=True, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final, cost_state_coeff=cost_state_coeff)
                key = jax.random.PRNGKey(111)
                key, subkey = jax.random.split(key)



                robot_sampled_states, robot_selected_states, control_signals, U = mppi(subkey, U, link_lengths, grab_goal, obstacle_position)
                control_signals = control_signals.reshape(4)
                # print('controls:', control_signals)
                
                # Update obstacle position
                obstacle_position += obstacle_velocity * dt
            else:
                if not grab_mode:
                    print("Grab Position Reached!")
                    grab_mode = True


                control_signals = np.array([0., 0., 0., grab_control])
                
                if link_lengths[-1] > 1.34 or link_lengths[-1] < 0.66:
                    if not grab_successful:
                        print("Grab Successful!")
                        grab_successful = True
                        # Store the relative position of the goal point to the grab point
                        relative_goal_position = grab_point - goal_point
                        print('relative_pos:', relative_goal_position)


                    # Apply control to bring the first three links to the desired range
                    else:
                        control_signals = np.array([-0.15, -0.15, -0.15, 0])

                        # Check if each link length is within the desired range
                        for i in range(3):
                            if 0.99 <= link_lengths[i] <= 1.00:
                                control_signals[i] = 0

                        # Update the goal point position relative to the updated grab point
                        goal_point = grab_point - relative_goal_position

                        


                        # Check if all link lengths are within the desired range
                        if control_signals[0] == 0 and control_signals[1] == 0 and control_signals[2] == 0:
                            # Freeze the video for an additional 0.5 second
                            print("Place Successful!")
                            for _ in range(int(0.5 / dt)):
                                fig.canvas.draw()
                                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                                writer.append_data(image[:, :, :3])
                                
                            break



        # Update link lengths using Euler integration

       
        link_lengths = integrate_link_lengths(link_lengths, control_signals, dt)

        total_time += dt


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

    writer.close()

    return success_count, collision_count, total_time

if __name__ == '__main__':


    # load the learned C-SDF model
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


    # create env for quantative statistics
    num_environments = 5
    num_trials = 1

    xlim_left = [-3.5, -1.2]
    # xlim_right = [-3.5, -1.2]


    #xlim_left = [2.46, 2.56]
    xlim_right = [2.4, 2.46]
    #ylim = [0.5, 3.8]
    ylim = [3.02, 3.1]
    #goal_xlim_left = [-2.5, -2.]
    goal_xlim_left = [2.0, 2.1]
    goal_xlim_right = [2.0, 2.1]
    goal_ylim = [1.0, 1.1]

    min_distance_obs = 0.8
    min_distance_goal = 1.2

    dt = 0.05

    #control_modes = ['clf_cbf', 'mppi']

    control_modes = ['mppi']

    #control_modes = ['random']

    for i in range(num_environments):
        obstacle_positions, obstacle_velocities, goal_point = generate_random_env(
            num_obstacles=3, xlim_left=xlim_left, xlim_right=xlim_right, ylim=ylim,
            goal_xlim_left=goal_xlim_left, goal_xlim_right=goal_xlim_right, goal_ylim=goal_ylim,
            min_distance_obs=min_distance_obs, min_distance_goal=min_distance_goal
        )

        for mode in control_modes:
            print(f"Running trials for control mode: {mode} in environment {i+1}")
            success_count = 0
            collision_count = 0
            total_time = 0.0

            for j in range(num_trials):
                trial_success, trial_collision, trial_time = main(jax_params, obstacle_positions, obstacle_velocities, goal_point, dt, mode, env_idx=i, trial_idx=j)
                success_count += trial_success
                collision_count += trial_collision
                total_time += trial_time

            success_rate = success_count / num_trials
            collision_rate = collision_count / num_trials
            avg_time = total_time / success_count if success_count > 0 else np.inf

            print(f"Control mode: {mode}")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Collision rate: {collision_rate:.2f}")
            print(f"Average time to reach goal: {avg_time:.2f} seconds")
            print("---")
