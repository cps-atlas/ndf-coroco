import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import jax
import jax.numpy as jnp

import time


from utils_3d import * 
from training.config_3D import *
from plot_utils import *

from control.mppi_functional import setup_mppi_controller

from robot_3D import Robot3D
from operate_env import Environment
from robot_config import * 
from evaluate_heatmap_3d import load_learned_csdf



def main(jax_params, wall_positions, obstacle_shapes, obstacle_points, goal_point, robot, dt, sphere_positions, sphere_radius, sphere_velocities, mode='mppi', env_idx=0, trial_idx=0):
    # Initialize the parameters for return
    total_time = 0.0

    # Create directory structure for saving videos
    result_dir = 'result_videos_new'
    env_dir = f'env{env_idx+1}'
    mode_dir = mode
    video_name = f'trial{trial_idx+1}.mp4'

    os.makedirs(os.path.join(result_dir, env_dir, mode_dir), exist_ok=True)
    video_name_x = f'trial{trial_idx+1}_x_view.mp4'
    video_name_y = f'trial{trial_idx+1}_y_view.mp4'

    video_path_x = os.path.join(result_dir, env_dir, mode_dir, video_name_x)
    video_path_y = os.path.join(result_dir, env_dir, mode_dir, video_name_y)

    writer_x = imageio.get_writer(video_path_x, fps=int(1/dt))
    writer_y = imageio.get_writer(video_path_y, fps=int(1/dt))


    # Set up MPPI controller
    # prediction_horizon need to be adaptive, based on the distance to goal. 
    # for 6-link: horizon = 4; 4,5-link: horizon = 20
    prediction_horizon = 18
    U = 0.0 * jnp.ones((prediction_horizon, 2 * robot.num_links))
    num_samples = 800
    costs_lambda = 0.03
    cost_goal_coeff = 15.0
    cost_safety_coeff = 1.1
    cost_goal_coeff_final = 18.0
    cost_safety_coeff_final = 1.1

    control_bound = 0.3

    cost_state_coeff = 10.0

    use_GPU = True


    mppi = setup_mppi_controller(learned_CSDF=jax_params, robot_n = 2 * robot.num_links, initial_horizon=prediction_horizon, samples=num_samples, input_size=2*robot.num_links, control_bound=control_bound, dt=dt, u_guess=None, use_GPU=use_GPU, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final, cost_state_coeff=cost_state_coeff)

    # Define the initial control signal
    if mode == 'random':
        control_signals = np.random.uniform(-0.12, 0.12, size=2 * robot.num_links)
    elif mode == 'mppi':
        control_signals = np.zeros(2 * robot.num_links)

    num_steps = 300
    goal_threshold = 0.3

    period = 10

    goal_distances = []
    estimated_obstacle_distances = []

    for step in range(num_steps):
        # Create a new figure and 3D axis for each frame
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the links
        legend_elements = plot_links_3d(robot.state, robot.link_radius, robot.link_length, ax)

        plot_env_3d(wall_positions, obstacle_shapes, goal_position, sphere_positions, sphere_radius, ax)

        # Set the plot limits and labels
        ax.set_xlim(-3, 9)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-2, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((4, 4, 4))

        # Add legend
        # ax.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='*', color='blue', linestyle='None', markersize=12, label='Goal')])

        plt.tight_layout()


        end_center, _, _ = compute_end_circle(robot.state, robot.link_radius, robot.link_length)

        # Calculate the distance between the end center and the goal point
        goal_distance = np.linalg.norm(end_center - goal_point)

        goal_distances.append(goal_distance)


        print('distance_to_goal:', goal_distance)

        # adaptive prediction horizon 
        if(goal_distance < 0.5):
            prediction_horizon = 5
            U = U[:prediction_horizon, :]   
            
            mppi = setup_mppi_controller(learned_CSDF=jax_params, robot_n = 2 * robot.num_links, initial_horizon=prediction_horizon, samples=num_samples, input_size=2*robot.num_links, control_bound=control_bound, dt=dt, u_guess=None, use_GPU=use_GPU, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final, cost_state_coeff=cost_state_coeff)

        # convert robot state (cable lengths) to configurations
        robot_config = state_to_config(robot.state, robot.link_radius, robot.link_length)

        #sdf_val, rbt_grad, sdf_grads = evaluate_model(jax_params, robot_config, robot.state, robot.link_radius, robot.link_length, env.obstacle_positions)

        #estimated_obstacle_distances.append(sdf_val)
        # print('estimated_obst_distances:', sdf_val)


        if mode == 'random':
            # Generate a random perturbation vector
            perturbation = np.random.uniform(-0.04, 0.04, size=2 * robot.num_links)

            # Update the control signal with the perturbation, clip the control signal to the desired range
            control_signals = np.clip(control_signals + perturbation, -0.12, 0.12)
        elif mode == 'mppi':
            key = jax.random.PRNGKey(step)
            key, subkey = jax.random.split(key)

            start_time = time.time()

            sphere_points = generate_sphere_points(sphere_positions, obst_radius=sphere_radius)

            all_obstacle_points = np.concatenate((obstacle_points, sphere_points), axis=0)

            robot_sampled_states, selected_robot_states, control_signals, U = mppi(subkey, U, robot.state.flatten(), goal_point, all_obstacle_points)

            print('time needed for MPPI:', time.time() - start_time)

            # Plot the trajectory of the end-effector along the selected states
            selected_end_effectors = []
            for i in range(selected_robot_states.shape[1]):
                
                robot_state = selected_robot_states[:,i].reshape(robot.num_links, 2)

                end_center, _, _ = compute_end_circle(robot_state, robot.link_radius, robot.link_length)


                selected_end_effectors.append(end_center)

            selected_end_effectors = np.array(selected_end_effectors)

            ax.plot(selected_end_effectors[:, 0], selected_end_effectors[:, 1], selected_end_effectors[:,2], 'b--', linewidth=2, label='Predicted End-Effector Trajectory')

        # Set the view to face the x-z plane (view from the x-axis)
        ax.view_init(elev=0, azim=-88)  # Set the view to face the x-z plane (view from the x-axis)
        fig.canvas.draw()
        buffer_x = fig.canvas.buffer_rgba()
        frame_x = np.asarray(buffer_x).reshape(buffer_x.shape[0], buffer_x.shape[1], 4)[:, :, :3]

        # Append the frames to the video writers
        writer_x.append_data(frame_x)

        if step == 0:
            for _ in range(int(0.5 / dt)):
                writer_x.append_data(frame_x)

        # Check if the goal is reached
        if goal_distance < goal_threshold:
            print("Goal Reached!")
            # Freeze the video for an additional 0.5 seconds
            for _ in range(int(0.8 / dt)):
                writer_x.append_data(frame_x)

        # Save the current frame for the y-view
        ax.view_init(elev=0, azim=2)  # Set the view to face the y-z plane (view from the y-axis)
        fig.canvas.draw()
        buffer_y = fig.canvas.buffer_rgba()
        frame_y = np.asarray(buffer_y).reshape(buffer_y.shape[0], buffer_y.shape[1], 4)[:, :, :3]


        writer_y.append_data(frame_y)

        # Freeze the video at the initial states for 0.5 seconds
        if step == 0:
            for _ in range(int(0.5 / dt)):
                writer_y.append_data(frame_y)

        # Check if the goal is reached
        if goal_distance < goal_threshold:
            print("Goal Reached!")
            # Freeze the video for an additional 0.5 seconds
            for _ in range(int(0.8 / dt)):
                writer_y.append_data(frame_y)
            break

        plt.close(fig)

        # Update the robot's edge lengths using the Robot3D instance
        robot.update_edge_lengths(control_signals, dt)

        total_time += dt

        t = total_time % period  # Time within the current period
        
        # Calculate the velocity based on the current time in the period
        if t < period / 2:
            # First half of the period: move in the initial direction
            obstacle_velocities = sphere_velocities
        else:
            # Second half of the period: move in the opposite direction
            obstacle_velocities = -sphere_velocities
        
        sphere_positions += obstacle_velocities * dt

    writer_x.close()
    writer_y.close()


    return goal_distances, estimated_obstacle_distances

if __name__ == '__main__':

    model_type = 'jax'

    #trained_model = "trained_models/torch_models_3d/eikonal_train.pth"
    #trained_model = "trained_models/torch_models_3d/test_1.pth"

    trained_model = "trained_models/torch_models_3d/eikonal_train_small.pth"

    net = load_learned_csdf(model_type, trained_model_path = trained_model)

    jax_params = net.params

    # create env for quantitative statistics
    corridor_pos = np.array([0, 0, -1])

    corridor_size = np.array([2, 2, 2])

    obstacle_positions = [
        np.array([2.5, 0, 4.8]),
        np.array([2.5, 0, 0]),
        np.array([1, -4, 3]), 
        np.array([5.3, 0, 2])
    ]
    obstacle_sizes = [
        np.array([1, 5, 3.2]),
        np.array([1, 5, 2]),
        np.array([3, 2, 4]),
        np.array([2., 5, 1])
    ]


    goal_position = np.array([4.8, 0, 3.6])

    # Generate dynamic spheres
    sphere_positions = np.array([
        np.array([4.0, -4.5, 3.6]),  # Sphere 1 position
        np.array([4.0, 0.0, 2.2])   # Sphere 2 position
    ])
    sphere_velocities = np.array([
        np.array([0.0, 1.0, 0.0]),  # Sphere 1 velocity
        np.array([0.0, 0.0, -0.8])  # Sphere 2 velocity
    ])

    sphere_radius = 0.5

    num_points_per_face = 40

    wall_positions, obstacle_shapes, obstacle_points = generate_realistic_env_3d(
        corridor_pos, corridor_size, obstacle_positions, obstacle_sizes, num_points_per_face=num_points_per_face)

    dt = 0.05
    control_modes = ['mppi']


    num_trials = 1


    for mode in control_modes:
        print(f"Running trials for control mode: {mode}")
        success_count = 0
        collision_count = 0
        total_time = 0.0

        for j in range(num_trials):
            # Create a Robot3D instance
            robot = Robot3D(num_links=NUM_OF_LINKS, link_radius=LINK_RADIUS, link_length=LINK_LENGTH)

            

            goal_distances, estimated_obstacle_distances = main(jax_params, wall_positions, obstacle_shapes, obstacle_points, goal_position, robot, dt, sphere_positions, sphere_radius, sphere_velocities, mode, env_idx=0, trial_idx=j)

            estimated_obstacle_distances = np.array(estimated_obstacle_distances)

            # Generate a unique name for the distance plot
            plot_name = f'env{1}_{mode}_trial{1}_distances.png'
            plot_path = os.path.join('distance_plots', plot_name)
            
            # Save the distance plot with the unique name
            plot_distances(goal_distances, estimated_obstacle_distances, dt, save_path=plot_path)
