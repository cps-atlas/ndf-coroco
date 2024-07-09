import pickle
import os
import numpy as np

import jax
from jax import jit
import jax.numpy as jnp

import torch
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.csdf_training_3D import train_3d, train_with_eikonal_3d, train_with_normal_loss_3d, train_jax_3d
from training.config_3D import *

from training_data.dataset import SoftRobotDataset, SoftRobotDataset_JAX

from torch.utils.data import DataLoader

from utils_3d import *
from robot_config import *

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

@jit
def evaluate_model(jax_params, rbt_configs, cable_lengths, link_radius, link_length, obstacle_points):
    # Predict signed distances
    @jit
    def apply_model(params, inputs):
        return CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).apply(params, inputs)

    # Ensure obstacle_points is a 2D array
    obstacle_points = jnp.array(obstacle_points, dtype=jnp.float32)
    if obstacle_points.ndim == 1:
        obstacle_points = obstacle_points.reshape(1, -1)

    rbt_configs = rbt_configs.reshape(NUM_OF_LINKS, 2)

    num_links = len(rbt_configs)
    num_points = obstacle_points.shape[0]

    # Compute the transformations using forward kinematics
    transformations = forward_kinematics(cable_lengths, link_radius, link_length)

    # Exclude the end-effector transformation
    transformations = transformations[:-1]

    # Initialize the minimum distance array and closest link index array
    min_distances = jnp.full(num_points, jnp.inf)
    closest_link_indices = jnp.zeros(num_points, dtype=jnp.int32)

    # Initialize the gradients arrays
    rbt_gradients = jnp.zeros((num_points, num_links * 2))
    obst_gradients = jnp.zeros((num_points, 3))

    for i in range(num_links):
        # Transform the points to the current link's local frame
        points_link = jnp.dot(jnp.linalg.inv(transformations[i]), jnp.hstack((obstacle_points, jnp.ones((num_points, 1)))).T).T[:, :3]

        # Prepare the input tensor for the current link
        inputs_link = jnp.hstack((jnp.repeat(rbt_configs[i].reshape(1, -1), num_points, axis=0), points_link))

        # Forward pass
        outputs_link = apply_model(jax_params, inputs_link)

        # Update the minimum distance array and closest link index array
        distances_link = outputs_link[:, 0]
        mask = distances_link < min_distances
        min_distances = jnp.where(mask, distances_link, min_distances)
        closest_link_indices = jnp.where(mask, i, closest_link_indices)

        # Compute the gradients using JAX automatic differentiation with vmap
        grad_fn = jax.vmap(jax.grad(lambda x: apply_model(jax_params, x)[0]), in_axes=0)
        gradients = grad_fn(inputs_link)

        # Update the obstacle gradients array
        obst_gradients = jnp.where(mask[:, None], gradients[:, 2:], obst_gradients)

    # Compute the robot gradients using chain rule
    # for i in range(num_links):
    #     # Prepare the input tensor for the current link
    #     points_link = jnp.dot(jnp.linalg.inv(transformations[i]), jnp.hstack((obstacle_points, jnp.ones((num_points, 1)))).T).T[:, :3]
    #     inputs_link = jnp.hstack((jnp.repeat(rbt_configs[i].reshape(1, -1), num_points, axis=0), points_link))

    #     # Compute the gradients using JAX automatic differentiation with vmap
    #     grad_fn = jax.vmap(jax.grad(lambda x: apply_model(jax_params, x)[0]), in_axes=0)
    #     gradients = grad_fn(inputs_link)

    #     # Update the robot gradients array based on the closest link
    #     mask_closest = closest_link_indices == i
    #     rbt_gradients = rbt_gradients.at[mask_closest, i*2:(i+1)*2].set(gradients[mask_closest, :2])

    #     # Apply chain rule for links before the closest link
    #     for j in range(i):
    #         # Compute the Jacobian matrix using JAX automatic differentiation
    #         jacobian_fn = jax.jacfwd(lambda x: jnp.dot(jnp.linalg.inv(transformations[j]), jnp.hstack((x, jnp.ones((num_points, 1)))).T).T[:, :3])
    #         jacobian = jacobian_fn(obstacle_points)

    #         # Update the robot gradients array using chain rule
    #         rbt_gradients = rbt_gradients.at[mask_closest, j*2:(j+1)*2].add(jnp.einsum('ijk,ik->ij', jacobian[mask_closest], gradients[mask_closest, 2:]))

    distances = min_distances

    return distances, rbt_gradients, obst_gradients

@jit
def compute_cbf_value_and_grad(jax_params, edge_lengths, obstacle_points, obstacle_velocities, link_radius, link_length):
    # Compute the distances and gradients from the robot to each obstacle point
    distances, rbt_gradients, obst_gradients = evaluate_model(jax_params, edge_lengths, obstacle_points, link_radius, link_length)

    # Ensure distances has the correct shape (num_obstacle_points,)
    assert distances.shape[0] == len(obstacle_points), "Mismatch in number of obstacle points"

    # Extract the minimum distance and corresponding gradients
    min_index = jnp.argmin(distances)
    cbf_h_val = distances[min_index]
    cbf_h_grad = rbt_gradients[min_index]

    # Compute the CBF time derivative (cbf_t_grad) for the closest obstacle
    cbf_t_grad = obst_gradients[min_index] @ obstacle_velocities[min_index]

    return cbf_h_val, cbf_h_grad, cbf_t_grad


'''
following is training with pytorch

'''

def main_torch(train_eikonal=False):
    # Load the saved dataset from the pickle file

    with open('training_data/dataset_3d_single_link_large.pickle', 'rb') as f:
        training_data = pickle.load(f)


    # Extract configurations, workspace points, and distances from the dataset
    configurations = np.array([entry['configuration'] for entry in training_data])
    points = np.array([entry['point'] for entry in training_data])
    distances = np.array([entry['distance'] for entry in training_data])
    #normals = np.array([entry['normal'] for entry in training_data])

    # Create dataset and data loaders
    dataset = SoftRobotDataset(configurations, points, distances)
    train_size = int(0.999 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    # Create neural network
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

    # training with pre-trained weights
    # net.load_state_dict(torch.load("trained_models/torch_models/trained_model_no_eikonal.pth"))

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")

    # Train the model
    if train_eikonal:
        print('training with eikonal start!')

        net = train_with_eikonal_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=1e-4, lambda_eikonal=0.02)
        # Save the trained model with Eikonal regularization
        torch.save(net.state_dict(), "trained_models/torch_models_3d/eikonal_train.pth")
    else:
        print('training start!')
        net = train_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=1e-4)
        #net = train_with_normal_loss(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=0.001, lambda_eikonal = 0.1)

        # Save the trained model with normal loss
        torch.save(net.state_dict(), "trained_models/torch_models_3d/test_1.pth")


'''
following is training with JAX

'''

def main_jax(train_eikonal=True):
    # Load the saved dataset from the pickle file
    with open('training_data/dataset_grid.pickle', 'rb') as f:
        training_data = pickle.load(f)
    
    # Extract configurations, workspace points, and distances from the dataset
    configurations = jnp.array([entry['configuration'] for entry in training_data])
    points = jnp.array([entry['point'] for entry in training_data])
    distances = jnp.array([entry['distances'] for entry in training_data])


    
    # Create dataset
    dataset = SoftRobotDataset_JAX(configurations, points, distances)

    # Create neural network
    net = CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)


    # Train the model
    if train_eikonal:
        trained_params = train_with_eikonal_3d(net, dataset, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, loss_threshold=0.1, lambda_eikonal=0.1)
        # Save the trained model with Eikonal regularization
        with open("trained_models/jax_models/trained_eikonal.pkl", "wb") as f:
            pickle.dump(trained_params, f)
    else:
        trained_params = train_jax_3d(net, dataset, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, loss_threshold=0.1)
        # Save the trained model with Eikonal regularization
        with open("trained_models/jax_models/trained_no_eikonal.pkl", "wb") as f:
            pickle.dump(trained_params, f)



if __name__ == "__main__":
    # Specify to Train with Eikonal or not 
    train_eikonal = True

    main_torch(train_eikonal)

    #main_jax(train_eikonal)
