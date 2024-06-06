import numpy as np
import torch
import matplotlib.pyplot as plt
from visualize_soft_link import plot_links
import imageio

from utils.csdf_net import CSDFNet
from training.config import *


from control.clf_qp import ClfQpController
from control.clf_cbf_qp import ClfCbfController


def evaluate_model(net, rbt_config, obstacle_point):
    # Prepare the input tensor
    configurations = torch.from_numpy(rbt_config).float().unsqueeze(0)
    points = torch.from_numpy(obstacle_point).float().unsqueeze(0)
    inputs = torch.cat((configurations, points), dim=1)
    inputs.requires_grad = True

    # Forward pa    net.eval()
    outputs = net(inputs)


    # Compute the gradients using back-propagation
    gradients = []
    for i in range(outputs.shape[1]):
        gradients.append(torch.autograd.grad(outputs[0, i], inputs, retain_graph=True)[0].unsqueeze(0))
    gradients = torch.cat(gradients, dim=0)


    # Extract the distances, gradients w.r.t robot configuration, and gradients w.r.t obstacle point
    distances = outputs.detach().numpy().flatten()

    
    link_gradients = gradients[:, :, :len(rbt_config)].detach().numpy()
    obst_gradients = gradients[:, :, len(rbt_config):].detach().numpy()

    return distances, link_gradients, obst_gradients

def compute_cbf_value_and_grad(net, rbt_config, obstacle_point, obstacle_velocity):
    # Compute the distances and gradients from each link to the obstacle point
    link_distances, link_gradients, obst_gradients = evaluate_model(net, rbt_config, obstacle_point)

    # Find the minimum distance across all links
    min_index = np.argmin(link_distances)


    cbf_h_val = link_distances[min_index]
    cbf_h_grad = link_gradients[min_index]

    # Compute the CBF time derivative (cbf_t_grad)
    cbf_t_grad = obst_gradients[min_index] @ obstacle_velocity

    return cbf_h_val, cbf_h_grad, cbf_t_grad


def integrate_link_lengths(link_lengths, control_signals, dt):
    # Euler integration to update link lengths
    link_lengths += control_signals * dt
    return link_lengths

def main(control_mode = 'clf_cbf'):
    # Define the number of links and time steps
    num_links = 4
    num_steps = 300
    dt = 0.02  # Time step size (in seconds)
    nominal_length = 1.0

    # Define the initial base points for the first link
    left_base = [-0.15, 0.0]
    right_base = [0.15, 0.0]

    # Initialize link lengths
    link_lengths = np.ones(num_links) * nominal_length

    # Define obstacle initial position and velocity
    obstacle_position = np.array([-2.0, 2.5])
    obstacle_velocity = np.array([1.0, 0.0])

    # Define the goal point for CLF-QP control
    goal_point = np.array([2.5, 2.5])

    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)
    net.load_state_dict(torch.load("trained_models/baseline_2.pth"))
    net.eval()

    # Create controllers for each control mode
    clf_cbf_controller = ClfCbfController()
    clf_qp_controller = ClfQpController()

    # Simulate and record videos for each control mode
    mode = control_mode
    
    link_lengths = np.ones(num_links) * nominal_length
    obstacle_position = np.array([-2.0, 2.5])

    # Create a video writer object for each control mode
    writer = imageio.get_writer(f"{mode}_animation.mp4", fps=int(1/dt))
    fig, ax = plt.subplots(figsize=(8, 8))

    for step in range(num_steps):
        if mode == 'clf_cbf':
            # Compute the CBF value and gradients
            cbf_h_val, cbf_h_grad, cbf_t_grad = compute_cbf_value_and_grad(net, link_lengths, obstacle_position, obstacle_velocity)
            # Generate control signals using the CBF-CLF controller
            control_signals = clf_cbf_controller.generate_controller(link_lengths, np.ones_like(link_lengths) * nominal_length, cbf_h_val, cbf_h_grad, cbf_t_grad)
            # Update obstacle position
            obstacle_position += obstacle_velocity * dt
        else:  # mode == 'clf_qp'
            # Compute the signed distance and gradient to the goal point
            sdf_val, sdf_grad, _ = evaluate_model(net, link_lengths, goal_point)
            # Generate control signals using the CLF-QP controller
            control_signals = clf_qp_controller.generate_controller(link_lengths, goal_point, sdf_val[-1], sdf_grad[-1])

        # Update link lengths using Euler integration
        link_lengths = integrate_link_lengths(link_lengths, control_signals, dt)

        # Clear the previous plot
        ax.clear()

        # Plot the links using the plot_links function
        plot_links(link_lengths, nominal_length, left_base, right_base, ax)

        if mode == 'clf_cbf':
            # Plot the obstacle
            ax.plot(obstacle_position[0], obstacle_position[1], 'ro', markersize=10)
        else:  # mode == 'clf_qp'
            # Plot the goal point
            ax.plot(goal_point[0], goal_point[1], 'go', markersize=10)

        # Convert the plot to an image and append it to the video
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        writer.append_data(image[:, :, :3])

    writer.close()

if __name__ == '__main__':
    control_mode = 'clf_qp'
    main(control_mode)