import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import jax
import jax.numpy as jnp

import time
import argparse

from utils_3d import * 
from training.config_3D import *
from plot_utils import *

from control.mppi_functional import setup_mppi_controller

from robot_3D import Robot3D
from robot_config import * 
from evaluate_heatmap import load_learned_csdf



def main(jax_params, wall_positions, obstacle_shapes, obstacle_points, goal_point, robot, dt, sphere_positions, sphere_radius, sphere_velocities, mode='mppi', env_idx=0, interactive_window = True):
    # Initialize the parameters for return
    total_time = 0.0

    # Create directory structure for saving videos
    result_dir = 'result_videos_cluttered'
    env_dir = f'env{env_idx+1}'
    mode_dir = mode

    os.makedirs(os.path.join(result_dir, env_dir, mode_dir), exist_ok=True)
    video_name = f'Link{NUM_OF_LINKS}.mp4'


    # Set up MPPI controller
    
    # for 6-link: horizon = 6; sample = 1000; 5-link: horizon = 24, sample = 800; 4-link: horizon = 18, 800 samples
    # prediction_horizon >= 2


    num_samples = 800
    costs_lambda = 0.02
    cost_goal_coeff = 12.0
    cost_safety_coeff = 1.1
    cost_goal_coeff_final = 22.0
    cost_safety_coeff_final = 1.1

    control_bound = 0.3

    cost_state_coeff = 50.0

    use_GPU = True

    prediction_horizon = 20

    # smaller horizon usually has better results for more links
    if NUM_OF_LINKS > 6:
        prediction_horizon = 7

    U = 0.0 * jnp.ones((prediction_horizon, 2 * robot.num_links))
    mppi = setup_mppi_controller(learned_CSDF=jax_params, robot_n=2 * robot.num_links, initial_horizon=prediction_horizon,
                                    samples=num_samples, input_size=2*robot.num_links, control_bound=control_bound,
                                    dt=dt, u_guess=U, use_GPU=use_GPU, costs_lambda=costs_lambda,
                                    cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff,
                                    cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final,
                                    cost_state_coeff=cost_state_coeff)

    # Define the initial control signal
    if mode == 'random':
        control_signals = np.random.uniform(-0.12, 0.12, size=2 * robot.num_links)

    num_steps = 300
    goal_threshold = 0.4

    period = 10

    goal_distances = []
    estimated_obstacle_distances = []

    # Create a new figure and 3D axis for each frame
    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')   

    frames = [] 
    

    for step in range(num_steps):
        ax.clear()

        # Plot the links
        legend_elements = plot_links_3d(robot.state, robot.link_radius, robot.link_length, ax)

        plot_env_3d(wall_positions, obstacle_shapes, goal_position, sphere_positions, sphere_radius, ax)


        end_center, _, _ = compute_end_circle(robot.state)

        # Calculate the distance between the end center and the goal point
        goal_distance = np.linalg.norm(end_center - goal_point)

        goal_distances.append(goal_distance)


        #print('distance_to_goal:', goal_distance)


        if mode == 'random':
            # Generate a random perturbation vector
            perturbation = np.random.uniform(-0.04, 0.04, size=2 * robot.num_links)

            # Update the control signal with the perturbation, clip the control signal to the desired range
            control_signals = np.clip(control_signals + perturbation, -0.12, 0.12)
        elif mode == 'mppi':

            key = jax.random.PRNGKey(step)
            key, subkey = jax.random.split(key)


            sphere_points = generate_sphere_points(sphere_positions, obst_radius=sphere_radius)

            all_obstacle_points = np.concatenate((obstacle_points, sphere_points), axis=0)

            # print('env_points_num:', all_obstacle_points.shape)

            # safety margin for point cloud data observations
            safety_margin = 0.1

            start_time = time.time()



            _, selected_robot_states, control_signals, U = mppi(subkey, U, robot.state.flatten(), goal_point, all_obstacle_points, safety_margin)

            print('time needed for MPPI:', time.time() - start_time)

            # Plot the trajectory of the end-effector along the selected states
            selected_end_effectors = []
            for i in range(selected_robot_states.shape[1]):
                
                robot_state = selected_robot_states[:,i].reshape(robot.num_links, 2)

                end_center, _, _ = compute_end_circle(robot_state)


                selected_end_effectors.append(end_center)

            selected_end_effectors = np.array(selected_end_effectors)

            ax.plot(selected_end_effectors[:, 0], selected_end_effectors[:, 1], selected_end_effectors[:,2], 'b--', linewidth=2, label='Predicted End-Effector Trajectory')

        # Set the plot limits and labels
        ax.set_xlim(-2, 8)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-2, 8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((4, 4, 4))

  
        # Redraw the plot
        fig.canvas.draw_idle()
        plt.tight_layout()
        # Capture the current plot image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        if interactive_window:
            plt.pause(0.01)  # Add a small pause to allow the plot to update



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

        # Check if the goal is reached
        if goal_distance < goal_threshold:
            print("Goal Reached!")
            print("time step needed:", step)
            # Append the last frame for the freeze duration
            freeze_duration = 0.8  # or 1 for 1 second
            fps = int(1 / dt)  # Determine the frame rate
            num_freeze_frames = int(freeze_duration * fps)
            last_frame = frames[-1]  # Get the last frame

            for _ in range(num_freeze_frames):
                frames.append(last_frame)
            break

    if interactive_window:
        plt.show()
    # remove the 1st frame
    video_path = os.path.join(result_dir, env_dir, mode_dir, video_name)
    imageio.mimsave(video_path, frames[1:], fps=int(1/dt))
    # imageio.mimsave(video_path, frames[1:], fps=int(50))

    return goal_distances, estimated_obstacle_distances

parser = argparse.ArgumentParser(description='Run the robot simulation.')
parser.add_argument('--no_interactive', dest='interactive_window', action='store_false', help='Disable interactive plotting')
parser.set_defaults(interactive_window=True)
args = parser.parse_args()

if __name__ == '__main__':

    model_type = 'jax'

    trained_model = "trained_models/torch_models_3d/eikonal_train_4_16.pth"

    #trained_model = "trained_models/torch_models_3d/grid_search_5_32.pth"

    net = load_learned_csdf(model_type, trained_model_path = trained_model)

    jax_params = net.params

    # create env for quantitative statistics
    corridor_pos = np.array([0, 0, -1])

    corridor_size = np.array([2, 2, 2])

    obstacle_positions = [
        np.array([2.5, 0, 5]),
        np.array([2.5, 0, 0]),
        np.array([5.3, 0, 2])
    ]
    obstacle_sizes = [
        np.array([1, 4.8, 3]),
        np.array([1, 5, 2]),
        np.array([2., 5, 1])
    ]


    goal_position = np.array([4.8, 0, 3.6])

    # Generate dynamic spheres
    sphere_positions = np.array([
        np.array([4.0, -4.5, 3.8]),  # Sphere 1 position
        np.array([4.0, 0.0, 2.2])   # Sphere 2 position
    ])
    sphere_velocities = np.array([
        np.array([0.0, 1.0, 0.0]),  # Sphere 1 velocity
        np.array([0.0, 0.0, -0.8])  # Sphere 2 velocity
    ])

    sphere_radius = 0.5

    num_points_per_area = 4

    wall_positions, obstacle_shapes, obstacle_points = generate_realistic_env_3d(
        corridor_pos, corridor_size, obstacle_positions, obstacle_sizes, num_points_per_unit_area=num_points_per_area)

    dt = 0.05
    control_modes = ['mppi']



    for mode in control_modes:
        print(f"Running trials for control mode: {mode}")

        # Create a Robot3D instance
        robot = Robot3D(num_links=NUM_OF_LINKS, link_radius=LINK_RADIUS, link_length=LINK_LENGTH)


        goal_distances, estimated_obstacle_distances = main(jax_params, wall_positions, obstacle_shapes, obstacle_points, goal_position, robot, dt, sphere_positions, sphere_radius, sphere_velocities, mode, env_idx=0, interactive_window=args.interactive_window)

