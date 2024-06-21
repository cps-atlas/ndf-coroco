import pickle
import numpy as np

import jax
from jax import jit
import jax.numpy as jnp

import torch
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.csdf_training import train, train_with_eikonal, train_with_normal_loss, train_jax
from training.config import *

from training_data.dataset import SoftRobotDataset, SoftRobotDataset_JAX

from torch.utils.data import DataLoader

@jit
def evaluate_model(jax_params, rbt_config, obstacle_points):
    # Predict signed distances
    @jit
    def apply_model(params, inputs):
        return CSDFNet_JAX(HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS).apply(params, inputs)

    # Ensure obstacle_points is a 2D array
    obstacle_points = jnp.array(obstacle_points, dtype=jnp.float32)
    if obstacle_points.ndim == 1:
        obstacle_points = obstacle_points.reshape(1, -1)

    # Prepare the input tensor
    configurations = jnp.repeat(jnp.array(rbt_config)[None, :], len(obstacle_points), axis=0)
    points = obstacle_points
    inputs = jnp.concatenate((configurations, points), axis=1)

    # Forward pass
    outputs = apply_model(jax_params, inputs)


    # Compute the gradients using JAX automatic differentiation
    gradients = []
    for i in range(len(obstacle_points)):
        point_gradients = []
        for j in range(outputs.shape[1]):  # Loop over the output dimension
            grad_fn = jax.grad(lambda x: apply_model(jax_params, x)[j])
            point_gradients.append(grad_fn(inputs[i]))
        gradients.append(jnp.array(point_gradients))
    gradients = jnp.array(gradients)


    # Extract the distances, gradients w.r.t robot configuration, and gradients w.r.t obstacle points
    link_gradients = gradients[:, :, :len(rbt_config)]
    obst_gradients = gradients[:, :, len(rbt_config):]

    

    return outputs, link_gradients, obst_gradients

@jit
def compute_cbf_value_and_grad(jax_params, rbt_config, obstacle_points, obstacle_velocities):
    # Compute the distances and gradients from each link to each obstacle point
    link_distances, link_gradients, obst_gradients = evaluate_model(jax_params, rbt_config, obstacle_points)



    # Ensure link_distances has the correct shape (num_obstacle_points, num_links)
    assert link_distances.shape[0] == len(obstacle_points), "Mismatch in number of obstacle points"
    assert link_distances.shape[1] == len(rbt_config), "Mismatch in number of links"

    # Find the minimum distance for each obstacle across all links
    min_indices = jnp.argmin(link_distances, axis=1)

    # Extract the minimum distances and corresponding gradients
    cbf_h_vals = link_distances[jnp.arange(len(obstacle_points)), min_indices]
    cbf_h_grads = link_gradients[jnp.arange(len(obstacle_points)), min_indices]

    # print('cbf_h_vals:', cbf_h_vals)
    # print('cbf_h_grads:', cbf_h_grads)

    # Compute the CBF time derivative (cbf_t_grad) for each obstacle
    cbf_t_grads = jnp.array([obst_gradients[i, min_indices[i]] @ obstacle_velocities[i] for i in range(len(obstacle_points))])


    return cbf_h_vals, cbf_h_grads, cbf_t_grads


'''
following are evaluations with torch net
'''

# def evaluate_model(net, rbt_config, obstacle_points):
#     # Ensure obstacle_points is a 2D array
#     obstacle_points = np.array(obstacle_points, dtype=np.float32)
#     if obstacle_points.ndim == 1:
#         obstacle_points = obstacle_points.reshape(1, -1)

#     # Prepare the input tensor
#     configurations = torch.from_numpy(rbt_config).float().unsqueeze(0).repeat(len(obstacle_points), 1)
#     points = torch.from_numpy(obstacle_points).float()
#     inputs = torch.cat((configurations, points), dim=1)
#     inputs.requires_grad = True

#     # Forward pass
#     net.eval()
#     outputs = net(inputs)

#     # Compute the gradients using back-propagation
#     gradients = []
#     for i in range(len(obstacle_points)):
#         point_gradients = []
#         for j in range(outputs.shape[1]):
#             net.zero_grad()
#             outputs[i, j].backward(retain_graph=True)
#             point_gradients.append(inputs.grad[i].numpy().copy())
#             inputs.grad.zero_()
#         gradients.append(np.array(point_gradients))
#     gradients = np.array(gradients)

#     # Extract the distances, gradients w.r.t robot configuration, and gradients w.r.t obstacle points
#     distances = outputs.detach().numpy()
#     link_gradients = gradients[:, :, :len(rbt_config)]
#     obst_gradients = gradients[:, :, len(rbt_config):]

#     # Convert numpy arrays back to JAX arrays
#     distances_jax = jnp.array(distances)
#     link_gradients_jax = jnp.array(link_gradients)
#     obst_gradients_jax = jnp.array(obst_gradients)


#     return distances_jax, link_gradients_jax, obst_gradients_jax

# def compute_cbf_value_and_grad(net, rbt_config, obstacle_points, obstacle_velocities):
#     # Compute the distances and gradients from each link to each obstacle point
#     link_distances, link_gradients, obst_gradients = evaluate_model(net, rbt_config, obstacle_points)

    
#     # Ensure link_distances has the correct shape (num_obstacle_points, num_links)
#     assert link_distances.shape[0] == len(obstacle_points), "Mismatch in number of obstacle points"
#     assert link_distances.shape[1] == len(rbt_config), "Mismatch in number of links"
    
#     # Find the minimum distance for each obstacle across all links
#     min_indices = np.argmin(link_distances, axis=1)

#     # Extract the minimum distances and corresponding gradients
#     cbf_h_vals = link_distances[np.arange(len(obstacle_points)), min_indices]
#     cbf_h_grads = link_gradients[np.arange(len(obstacle_points)), min_indices]

#     # print('cbf_h_vals:', cbf_h_vals)
#     # print('cbf_h_grads:', cbf_h_grads)

#     # Compute the CBF time derivative (cbf_t_grad) for each obstacle
#     cbf_t_grads = []
#     for i in range(len(obstacle_points)):
#         cbf_t_grad = obst_gradients[i, min_indices[i]] @ obstacle_velocities[i]
#         cbf_t_grads.append(cbf_t_grad)
#     cbf_t_grads = np.array(cbf_t_grads)

#     return cbf_h_vals, cbf_h_grads, cbf_t_grads


'''
following is training with pytorch

'''

def main_torch(train_eikonal=False):
    # Load the saved dataset from the pickle file
    with open('training_data/dataset_grid.pickle', 'rb') as f:
        training_data = pickle.load(f)

    # Extract configurations, workspace points, and distances from the dataset
    configurations = np.array([entry['configurations'] for entry in training_data])
    points = np.array([entry['point'] for entry in training_data])
    distances = np.array([entry['distances'] for entry in training_data])
    #normals = np.array([entry['normal'] for entry in training_data])

    # Create dataset and data loaders
    dataset = SoftRobotDataset(configurations, points, distances)
    train_size = int(0.999 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    # Create neural network
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)

    # training with pre-trained weights
    # net.load_state_dict(torch.load("trained_models/torch_models/trained_model_no_eikonal.pth"))

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    # Train the model
    if train_eikonal:
        net = train_with_eikonal(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=0.001, lambda_eikonal=0.1)
        # Save the trained model with Eikonal regularization
        torch.save(net.state_dict(), "trained_models/torch_models/new_test.pth")
    else:
        net = train(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=0.001)
        #net = train_with_normal_loss(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=0.001, lambda_eikonal = 0.1)

        # Save the trained model with normal loss
        torch.save(net.state_dict(), "trained_models/torch_models/new_test.pth")


'''
following is training with JAX

'''

def main_jax(train_eikonal=True):
    # Load the saved dataset from the pickle file
    with open('training_data/dataset_grid.pickle', 'rb') as f:
        training_data = pickle.load(f)
    
    # Extract configurations, workspace points, and distances from the dataset
    configurations = jnp.array([entry['configurations'] for entry in training_data])
    points = jnp.array([entry['point'] for entry in training_data])
    distances = jnp.array([entry['distances'] for entry in training_data])


    
    # Create dataset
    dataset = SoftRobotDataset_JAX(configurations, points, distances)

    # Create neural network
    net = CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)


    # Train the model
    if train_eikonal:
        trained_params = train_with_eikonal(net, dataset, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, loss_threshold=0.1, lambda_eikonal=0.1)
        # Save the trained model with Eikonal regularization
        with open("trained_models/jax_models/trained_eikonal.pkl", "wb") as f:
            pickle.dump(trained_params, f)
    else:
        trained_params = train_jax(net, dataset, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, loss_threshold=0.1)
        # Save the trained model with Eikonal regularization
        with open("trained_models/jax_models/trained_no_eikonal.pkl", "wb") as f:
            pickle.dump(trained_params, f)



if __name__ == "__main__":
    # Specify to Train with Eikonal or not 
    train_eikonal = False

    main_torch(train_eikonal)

    #main_jax(train_eikonal)
