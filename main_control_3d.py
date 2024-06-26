import os

import numpy as np
import matplotlib.pyplot as plt

import imageio

from utils_3d import * 


import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


from control.mppi_functional import setup_mppi_controller

from flax.core import freeze

from robot_3D import Robot3D
from operate_env import Environment

def main(env, robot, dt, mode='random', env_idx=0, trial_idx=0):
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
    prediction_horizon = 10
    U = 0.0 * jnp.ones((prediction_horizon, 2 * robot.num_links))
    num_samples = 5
    costs_lambda = 0.03
    cost_goal_coeff = 18.0
    cost_safety_coeff = 2.2
    cost_goal_coeff_final = 15.0
    cost_safety_coeff_final = 1.8

    control_bound = 0.1

    cost_state_coeff = 10.0

    use_GPU = True

    mppi = setup_mppi_controller(learned_CSDF=None, horizon=prediction_horizon, samples=num_samples, input_size=2*robot.num_links, control_bound=control_bound, dt=dt, u_guess=None, use_GPU=use_GPU, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, cost_goal_coeff_final=cost_goal_coeff_final, cost_safety_coeff_final=cost_safety_coeff_final, cost_state_coeff=cost_state_coeff)

    # Define the initial control signal
    if mode == 'random':
        control_signals = np.random.uniform(-0.12, 0.12, size=2 * robot.num_links)
    elif mode == 'mppi':
        control_signals = np.zeros(2 * robot.num_links)

    num_steps = 100

    for step in range(num_steps):
        # Create a new figure and 3D axis for each frame
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the links
        legend_elements = plot_links_3d(robot.state, robot.link_radius, robot.link_length, ax)

        # Plot the goal point
        ax.plot([env.goal_point[0]], [env.goal_point[1]], [env.goal_point[2]], marker='*', markersize=12, color='blue', label='Goal')

        # Set the plot limits and labels
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((4, 4, 4))

        # Add legend
        ax.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='*', color='blue', linestyle='None', markersize=12, label='Goal')], loc='upper right')

        plt.tight_layout()


        end_center, _, _ = compute_end_circle(robot.state, robot.link_radius, robot.link_length)

        # Calculate the distance between the end center and the goal point
        goal_distance = np.linalg.norm(end_center - env.goal_point)

        print('Goal distance:', goal_distance)

        if mode == 'random':
            # Generate a random perturbation vector
            perturbation = np.random.uniform(-0.04, 0.04, size=2 * robot.num_links)

            # Update the control signal with the perturbation, clip the control signal to the desired range
            control_signals = np.clip(control_signals + perturbation, -0.12, 0.12)
        elif mode == 'mppi':
            key = jax.random.PRNGKey(step)
            key, subkey = jax.random.split(key)

            robot_sampled_states, selected_robot_states, control_signals, U = mppi(subkey, U, robot.state.flatten(), env.goal_point, env.obstacle_positions)

            # Plot the trajectory of the end-effector along the selected states
            selected_end_effectors = []
            for i in range(selected_robot_states.shape[1]):
                
                robot_state = selected_robot_states[:,i].reshape(4,2)

                end_center, _, _ = compute_end_circle(robot_state, robot.link_radius, robot.link_length)


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



        # Update the robot's edge lengths using the Robot3D instance
        robot.update_edge_lengths(control_signals, dt)

        total_time += dt

        # Check if the goal is reached
        if goal_distance < 0.1:
            print("Goal Reached!")
            success_count = 1
            # Freeze the video for an additional 0.5 seconds
            for _ in range(int(0.5 / dt)):
                writer.append_data(frame)
            break

    writer.close()

    return success_count, collision_count, total_time

if __name__ == '__main__':
    # create env for quantitative statistics
    num_environments = 5
    num_trials = 1
    xlim = [-4.5, 4.5]
    ylim = [-4.5, 4.5]
    zlim = [0.0, 4.0]

    goal_xlim = [2.0, 3.0]
    goal_ylim = [2.0, 3.0]
    goal_zlim = [1.0, 2.0]

    min_distance_obs = 0.8
    min_distance_goal = 1.2

    dt = 0.05
    control_modes = ['mppi']

    obst_radius = 0.4

    for i in range(num_environments):
        obstacle_positions, obstacle_velocities, goal_point = generate_random_env_3d(
            num_obstacles=3, xlim=xlim, ylim=ylim, zlim=zlim,
            goal_xlim=goal_xlim, goal_ylim=goal_ylim, goal_zlim=goal_zlim,
            min_distance_obs=min_distance_obs, min_distance_goal=min_distance_goal
        )

        for mode in control_modes:
            print(f"Running trials for control mode: {mode} in environment {i+1}")
            success_count = 0
            collision_count = 0
            total_time = 0.0

            for j in range(num_trials):
                # Create a Robot3D instance
                robot = Robot3D(num_links=4, link_radius=0.15, link_length=1.0)

                # Create the operation environment
                env = Environment(obstacle_positions=obstacle_positions, obstacle_velocities=obstacle_velocities, obst_radius=obst_radius, goal_point=goal_point)

                trial_success, trial_collision, trial_time = main(env, robot, dt, mode, env_idx=i, trial_idx=j)
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