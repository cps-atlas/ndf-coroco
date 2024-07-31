import pickle
import numpy as np


import jax.numpy as jnp

import torch
from network.csdf_net import CSDFNet, CSDFNet_JAX
from training.csdf_training_3D import train_3d, train_with_eikonal_3d
from training.config_3D import *

from training_data.dataset import SoftRobotDataset, SoftRobotDataset_JAX

from torch.utils.data import DataLoader

from itertools import product


'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')



def grid_search(train_dataset, val_dataset, device):
    # Define the hyperparameter search space
    hidden_sizes = [16, 24, 32]
    num_layers = [2, 3, 4]
    # learning_rates = [0.001, 0.003, 0.005, 0.01]
    # batch_sizes = [128, 256]

    learning_rates = [0.003]
    batch_sizes = [256]

    best_val_loss = float('inf')
    best_hyperparams = None
    best_model = None

    # Perform grid search
    for hidden_size, num_layer, learning_rate, batch_size in product(hidden_sizes, num_layers, learning_rates, batch_sizes):
        print(f"Training with hidden_size={hidden_size}, num_layers={num_layer}, learning_rate={learning_rate}, batch_size={batch_size}")

        # Create data loaders with the current batch size
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size)

        # Create a new instance of the model with the current hyperparameters
        net = CSDFNet(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layer)

        # Train and evaluate the model
        net, val_loss = train_with_eikonal_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.05)

        print('resulting loss:', val_loss)

        # Update the best hyperparameters and model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparams = (hidden_size, num_layer, learning_rate, batch_size)
            best_model = net

    print(f"Best hyperparameters: hidden_size={best_hyperparams[0]}, num_layers={best_hyperparams[1]}, learning_rate={best_hyperparams[2]}, batch_size={best_hyperparams[3]}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return best_model, best_hyperparams


'''
following is training with pytorch

'''

def main_torch(train_eikonal=False):
    # Load the saved dataset from the pickle file

    with open('training_data/dataset_3d_single_link.pickle', 'rb') as f:
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
    train_dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)   # train with all data
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    # Create neural network
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    if train_eikonal:
        print('training with eikonal start!')
        best_model, best_hyperparams = grid_search(train_dataset, val_dataset, device)

        # net, _ = train_with_eikonal_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=1e-4, lambda_eikonal=0.05)
        # Save the trained model with Eikonal regularization
        # torch.save(net.state_dict(), "trained_models/torch_models_3d/eikonal_train_4_16_new.pth")
    else:
        print('training without eikonal start!')
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
