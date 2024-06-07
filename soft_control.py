import numpy as np
import torch
import matplotlib.pyplot as plt
from visualize_soft_link import plot_links
import imageio

from utils.csdf_net import CSDFNet
from training.config import *


from control.clf_qp import ClfQpController
from control.clf_cbf_qp import ClfCbfController

def evaluate_model_sdf(net, rbt_config, obstacle_point):
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
        gradients.append(torch.autograd.grad(outputs[0, i], inputs, retain_graph=True)[0])
    gradients = torch.cat(gradients, dim=0)


    # Extract the distances, gradients w.r.t robot configuration, and gradients w.r.t obstacle point
    distances = outputs.detach().numpy().flatten()

    
    link_gradients = gradients[ :, :len(rbt_config)].detach().numpy()
    obst_gradients = gradients[ :, len(rbt_config):].detach().numpy()

    print('distance:', distances)
    print('gradient:', gradients)

    return distances, link_gradients, obst_gradients



def evaluate_model(net, rbt_config, obstacle_points):
    # Ensure obstacle_points is a 2D array
    obstacle_points = np.array(obstacle_points, dtype=np.float32)
    if obstacle_points.ndim == 1:
        obstacle_points = obstacle_points.reshape(1, -1)

    # Prepare the input tensor
    configurations = torch.from_numpy(rbt_config).float().unsqueeze(0).repeat(len(obstacle_points), 1)
    points = torch.from_numpy(obstacle_points).float()
    inputs = torch.cat((configurations, points), dim=1)
    inputs.requires_grad = True

    # Forward pass
    net.eval()
    outputs = net(inputs)

    # Compute the gradients using back-propagation
    gradients = []
    for i in range(len(obstacle_points)):
        point_gradients = []
        for j in range(outputs.shape[1]):
            net.zero_grad()
            outputs[i, j].backward(retain_graph=True)
            point_gradients.append(inputs.grad[i].numpy().copy())
            inputs.grad.zero_()
        gradients.append(np.array(point_gradients))
    gradients = np.array(gradients)

    # Extract the distances, gradients w.r.t robot configuration, and gradients w.r.t obstacle points
    distances = outputs.detach().numpy()
    link_gradients = gradients[:, :, :len(rbt_config)]
    obst_gradients = gradients[:, :, len(rbt_config):]




    return distances, link_gradients, obst_gradients

# def compute_cbf_value_and_grad(net, rbt_config, obstacle_point, obstacle_velocity):
#     # Compute the distances and gradients from each link to the obstacle point
#     link_distances, link_gradients, obst_gradients = evaluate_model(net, rbt_config, obstacle_point)

#     # Find the minimum distance across all links
#     min_index = np.argmin(link_distances)


#     cbf_h_val = link_distances[min_index]
#     cbf_h_grad = link_gradients[min_index]

#     # Compute the CBF time derivative (cbf_t_grad)
#     cbf_t_grad = obst_gradients[min_index] @ obstacle_velocity

#     return cbf_h_val, cbf_h_grad, cbf_t_grad

def compute_cbf_value_and_grad(net, rbt_config, obstacle_points, obstacle_velocities):
    # Compute the distances and gradients from each link to each obstacle point
    link_distances, link_gradients, obst_gradients = evaluate_model(net, rbt_config, obstacle_points)

    
    # Ensure link_distances has the correct shape (num_obstacle_points, num_links)
    assert link_distances.shape[0] == len(obstacle_points), "Mismatch in number of obstacle points"
    assert link_distances.shape[1] == len(rbt_config), "Mismatch in number of links"
    
    # Find the minimum distance for each obstacle across all links
    min_indices = np.argmin(link_distances, axis=1)

    # Extract the minimum distances and corresponding gradients
    cbf_h_vals = link_distances[np.arange(len(obstacle_points)), min_indices]
    cbf_h_grads = link_gradients[np.arange(len(obstacle_points)), min_indices]

    # print('cbf_h_vals:', cbf_h_vals)
    # print('cbf_h_grads:', cbf_h_grads)

    # Compute the CBF time derivative (cbf_t_grad) for each obstacle
    cbf_t_grads = []
    for i in range(len(obstacle_points)):
        cbf_t_grad = obst_gradients[i, min_indices[i]] @ obstacle_velocities[i]
        cbf_t_grads.append(cbf_t_grad)
    cbf_t_grads = np.array(cbf_t_grads)

    return cbf_h_vals, cbf_h_grads, cbf_t_grads



def integrate_link_lengths(link_lengths, control_signals, dt):
    # Euler integration to update link lengths
    link_lengths += control_signals * dt
    return link_lengths

def main(net, obstacle_position, obstacle_velocity, goal_point, dt, control_mode = 'clf_cbf'):
    # Define the number of links and time steps
    num_links = 4
    num_steps = 400
    
    nominal_length = 1.0

    # Define the initial base points for the first link
    left_base = [-0.15, 0.0]
    right_base = [0.15, 0.0]

    # Initialize link lengths
    link_lengths = np.ones(num_links) * nominal_length


    net.eval()

    # Create controllers for each control mode
    clf_cbf_controller = ClfCbfController()
    clf_qp_controller = ClfQpController()

    # Simulate and record videos for each control mode
    mode = control_mode
    
    link_lengths = np.ones(num_links) * nominal_length

    # Create a video writer object for each control mode
    writer = imageio.get_writer(f"{mode}_animation.mp4", fps=int(1/dt))
    fig, ax = plt.subplots(figsize=(8, 8))

    for step in range(num_steps):

        # Compute the signed distance and gradient to the goal point
        sdf_val, sdf_grad, _ = evaluate_model(net, link_lengths, goal_point)

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
            cbf_h_val, cbf_h_grad, cbf_t_grad = compute_cbf_value_and_grad(net, link_lengths, obstacle_position, obstacle_velocity)



            # Generate control signals using the CBF-CLF controller
            control_signals = clf_cbf_controller.generate_controller(link_lengths, cbf_h_val, cbf_h_grad, cbf_t_grad, sdf_val[-1][-1], sdf_grad[-1][-1])

            # Update obstacle position
            
            obstacle_position += obstacle_velocity * dt

        else:  # mode == 'clf_qp'


            # Generate control signals using the CLF-QP controller
            control_signals = clf_qp_controller.generate_controller(link_lengths, sdf_val[-1][-1], sdf_grad[-1][-1])

        # Update link lengths using Euler integration
        link_lengths = integrate_link_lengths(link_lengths, control_signals, dt)

        # Clear the previous plot
        ax.clear()

        # Plot the links using the plot_links function
        legend_elements = plot_links(link_lengths, nominal_length, left_base, right_base, ax)

        if mode == 'clf_cbf':
            
            goal_plot, = ax.plot(goal_point[0]-0.1, goal_point[1]+0.1, marker='*', markersize=15, color='blue', label = 'Goal')
            legend_elements.append(goal_plot)
            # Plot the obstacles
            obstacle_plot = None
            for i, obstacle in enumerate(obstacle_position):
                if i == 0:
                    obstacle_plot, = ax.plot(obstacle[0], obstacle[1], 'ro', markersize=10, label='Obstacles')
                else:
                    ax.plot(obstacle[0], obstacle[1], 'ro', markersize=10)
            if obstacle_plot is not None:
                legend_elements.append(obstacle_plot)
        
        else:  # mode == 'clf_qp'
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
        [2.4, 3.4],
        [-1.0, 2.2],
        [1.1, 0.4]
    ])
    obstacle_velocities = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ])

    # Define the goal point for CLF-QP control
    goal_point = np.array([3.3, 0.8])

    # time step 
    dt = 0.02 

    control_mode = 'clf_cbf'


    # load the learned C-SDF model
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)
    net.load_state_dict(torch.load("trained_models/baseline_3_more_data.pth"))

    # simulate the control performance
    main(net, obstacle_positions, obstacle_velocities, goal_point, dt,control_mode)