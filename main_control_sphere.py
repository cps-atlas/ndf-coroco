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
from evaluate_heatmap import load_learned_csdf


def main(jax_params, env, robot, dt, mode='random', env_idx=0, trial_idx=0):
    # Initialize the parameters for return
    success_count = 0
    collision_count = 0
    total_time = 0.0

    # Create directory structure for saving videos
    result_dir = 'result_videos_3d'
    env_dir = f'env{env_idx+1}'
    mode_dir = mode
    video_name = f'trial{trial_idx+1}.mp4'

    os.makedirs(os.path.join(result_dir, env_dir, mode_dir), exist_ok=True)
    video_path = os.path.join(result_dir, env_dir, mode_dir, video_name)

    # Create a video writer object
    writer = imageio.get_writer(video_path, fps=int(1/dt))

    # Set up MPPI controller
    prediction_horizon = 20
    U = 0.0 * jnp.ones((prediction_horizon, 2 * robot.num_links))
    num_samples = 2000
    costs_lambda = 0.03
    cost_goal_coeff = 15.0
    cost_safety_coeff = 1.1
    cost_goal_coeff_final = 18.0
    cost_safety_coeff_final = 1.1

    control_bound = 0.3

    cost_state_coeff = 100.0

    use_GPU = True

    mppi = setup_mppi_controller(learned_CSDF=jax_params, initial_horizon=prediction_horizon, samples=num_samples, input_size=2*robot.num_links, control_bound=control_bound, dt=dt, u_guess=None, use_GPU=use_GPU, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final, cost_state_coeff=cost_state_coeff)

    # Define the initial control signal
    if mode == 'random':
        control_signals = np.random.uniform(-0.12, 0.12, size=2 * robot.num_links)
    elif mode == 'mppi':
        control_signals = np.zeros(2 * robot.num_links)

    num_steps = 200

    goal_threshold = 0.15

    goal_distances = []
    estimated_obstacle_distances = []

    for step in range(num_steps):
        # Create a new figure and 3D axis for each frame
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the links
        legend_elements = plot_links_3d(robot.state, robot.link_radius, robot.link_length, ax)

        plot_random_env_3d(env.obstacle_positions, env.goal_point, env.obst_radius, ax)

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


        end_center, _, _ = compute_end_circle(robot.state)

        # Calculate the distance between the end center and the goal point
        goal_distance = np.linalg.norm(end_center - env.goal_point)

        goal_distances.append(goal_distance)

        print('distance_to_goal:', goal_distance)


        # convert robot state (cable lengths) to configurations
        robot_config = state_to_config(robot.state)

        sdf_val = evaluate_model(jax_params, robot.state, env.obstacle_positions)

        estimated_obstacle_distances.append(sdf_val)



        if mode == 'random':
            # Generate a random perturbation vector
            perturbation = np.random.uniform(-0.04, 0.04, size=2 * robot.num_links)

            # Update the control signal with the perturbation, clip the control signal to the desired range
            control_signals = np.clip(control_signals + perturbation, -0.12, 0.12)
        elif mode == 'mppi':
            key = jax.random.PRNGKey(step)
            key, subkey = jax.random.split(key)

            # safety margin for sphere dynamical obstacles
            safety_margin = 0.1 + env.obst_radius

            start_time = time.time()

            robot_sampled_states, selected_robot_states, control_signals, U = mppi(subkey, U, robot.state.flatten(), env.goal_point, env.obstacle_positions, safety_margin)

            # print('time needed for MPPI:', time.time() - start_time)

            # Plot the trajectory of the end-effector along the selected states
            selected_end_effectors = []
            for i in range(selected_robot_states.shape[1]):
                
                robot_state = selected_robot_states[:,i].reshape(4,2)

                end_center, _, _ = compute_end_circle(robot_state)


                selected_end_effectors.append(end_center)

            selected_end_effectors = np.array(selected_end_effectors)

            ax.plot(selected_end_effectors[:, 0], selected_end_effectors[:, 1], selected_end_effectors[:,2], 'b--', linewidth=2, label='Predicted End-Effector Trajectory')



        # Save the current frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame)

        # Freeze the video at the initial states for 0.5 seconds
        if step == 0:
            for _ in range(int(0.5 / dt)):
                writer.append_data(frame)

        plt.close(fig)


        # print('control:', control_signals)

        # Update the robot's edge lengths using the Robot3D instance
        robot.update_edge_lengths(control_signals, dt)

        # Update the environment
        env.update_obstacles(dt)

        total_time += dt

        # Check if the goal is reached
        if goal_distance < goal_threshold:
            print("Goal Reached!")
            success_count = 1
            # Freeze the video for an additional 0.5 seconds
            for _ in range(int(0.8 / dt)):
                writer.append_data(frame)
            break

    writer.close()

    return success_count, collision_count, total_time, goal_distances, estimated_obstacle_distances

if __name__ == '__main__':

    model_type = 'jax'

    #trained_model = "trained_models/torch_models_3d/eikonal_train.pth"
    #trained_model = "trained_models/torch_models_3d/test_1.pth"

    trained_model = "trained_models/torch_models_3d/eikonal_train_4_16.pth"

    net = load_learned_csdf(model_type, trained_model_path = trained_model)

    jax_params = net.params

    # create env for quantitative statistics
    num_environments = 1
    num_trials = 1
    xlim = [-4, 4]
    ylim = [-4, 4]
    zlim = [-1, 6.0]

    goal_xlim = [-4.0, 4.0]
    goal_ylim = [-4.0, 4.0]
    goal_zlim = [-1.0, 6.0]

    min_distance_obs = 1.5
    min_distance_goal = 2.0

    dt = 0.05
    control_modes = ['mppi']

    obst_radius = 0.7

    # Create a directory to store the distance plots
    os.makedirs('distance_plots', exist_ok=True)

    # Create a Robot3D instance
    initial_rbt_state = jnp.ones((NUM_OF_LINKS, 2)) * LINK_LENGTH

    for i in range(num_environments):

        # make sure there is no initial collision with the generated random env
        while True:
            obstacle_positions, obstacle_velocities, goal_point = generate_random_env_3d(
                num_obstacles=8, xlim=xlim, ylim=ylim, zlim=zlim,
                goal_xlim=goal_xlim, goal_ylim=goal_ylim, goal_zlim=goal_zlim,
                min_distance_obs=min_distance_obs, min_distance_goal=min_distance_goal
            )
            sdf_val = evaluate_model(jax_params, initial_rbt_state, obstacle_positions)
            min_sdf = np.min(sdf_val)
            
            if min_sdf > obst_radius + 0.1:
                print(f'Environment {i+1}: Valid. Minimum SDF: {min_sdf:.4f}')
                break
            else:
                print(f'Environment {i+1}: Invalid. Minimum SDF: {min_sdf:.4f}. Regenerating...')
        


        for mode in control_modes:
            print(f"Running trials for control mode: {mode} in environment {i+1}")
            success_count = 0
            collision_count = 0
            total_time = 0.0

            for j in range(num_trials):
                # Create a Robot3D instance
                robot = Robot3D(num_links=NUM_OF_LINKS, link_radius=LINK_RADIUS, link_length=LINK_LENGTH)

                # Create the operation environment
                env = Environment(obstacle_positions=obstacle_positions, obstacle_velocities=obstacle_velocities, obst_radius=obst_radius, goal_point=goal_point)

                trial_success, trial_collision, trial_time, goal_distances, estimated_obstacle_distances = main(jax_params, env, robot, dt, mode, env_idx=i, trial_idx=j)
                success_count += trial_success
                collision_count += trial_collision
                total_time += trial_time

                estimated_obstacle_distances = np.array(estimated_obstacle_distances)

                # Generate a unique name for the distance plot
                plot_name = f'env{i+1}_{mode}_trial{j+1}_distances.png'
                plot_path = os.path.join('distance_plots', plot_name)
                
                # Save the distance plot with the unique name
                plot_distances(goal_distances, estimated_obstacle_distances, dt, save_path=plot_path)

            success_rate = success_count / num_trials
            collision_rate = collision_count / num_trials
            avg_time = total_time / success_count if success_count > 0 else np.inf

            print(f"Control mode: {mode}")
            print(f"Success rate: {success_rate:.2f}")
            print(f"Collision rate: {collision_rate:.2f}")
            print(f"Average time to reach goal: {avg_time:.2f} seconds")
            print("---")