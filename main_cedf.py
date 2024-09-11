import pickle
import numpy as np


import torch
from network.csdf_net import CSDFNet
from training.csdf_training_3D import train_3d, train_with_eikonal_3d, train_eikonal_moe
from training.config_3D import *

from training_data.dataset import SoftRobotDataset

from torch.utils.data import DataLoader

from itertools import product


'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')



def grid_search(train_dataloader, val_dataloader, device):
    # Define the hyperparameter search space
    hidden_sizes = [16, 24, 32]
    num_layers = [2, 3, 4, 5]
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


        # Create a new instance of the model with the current hyperparameters
        net = CSDFNet(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layer)

        # Train and evaluate the model
        # net, val_loss = train_with_eikonal_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.05)

        net, val_loss = train_eikonal_moe(net, train_dataloader, val_dataloader, NUM_EPOCHS, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.05, lambda_moe=2.0)

        print('resulting loss:', val_loss)

        # Save the trained model with the current hyperparameters
        model_name = f"grid_search_moe_{num_layer}_{hidden_size}.pth"
        torch.save(net.state_dict(), f"trained_models/torch_models_3d/{model_name}")

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

    with open('training_data/dataset_3d_single_link_train.pickle', 'rb') as f:
        training_data = pickle.load(f)


    # Extract configurations, workspace points, and distances from the dataset
    configurations = np.array([entry['configuration'] for entry in training_data])
    points = np.array([entry['point'] for entry in training_data])
    distances = np.array([entry['distance'] for entry in training_data])
    #normals = np.array([entry['normal'] for entry in training_data])

    # Create dataset and data loaders
    dataset = SoftRobotDataset(configurations, points, distances)
    train_dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)  

    # load the validation dataset
    with open('training_data/dataset_3d_single_link_validation.pickle', 'rb') as f:
        validation_data = pickle.load(f)


    # Extract configurations, workspace points, and distances from the dataset
    configurations = np.array([entry['configuration'] for entry in validation_data])
    points = np.array([entry['point'] for entry in validation_data])
    distances = np.array([entry['distance'] for entry in validation_data])

    val_dataset = SoftRobotDataset(configurations, points, distances)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    # Create neural network
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    if train_eikonal:
        print('training with eikonal start!')
        best_model, best_hyperparams = grid_search(train_dataloader, val_dataloader, device)

        # net, _ = train_with_eikonal_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=1e-4, lambda_eikonal=0.05)
        # net, _ = train_eikonal_moe(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=1e-4, lambda_eikonal=0.05, lambda_moe=2.0)
        # Save the trained model with Eikonal regularization
        # torch.save(net.state_dict(), "trained_models/torch_models_3d/eikonal_moe_train.pth")
    else:
        print('training without eikonal start!')
        net = train_3d(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=1e-4)

        # Save the trained model with normal loss
        torch.save(net.state_dict(), "trained_models/torch_models_3d/test_1.pth")


if __name__ == "__main__":
    # Specify to Train with Eikonal or not 
    train_eikonal = True

    main_torch(train_eikonal)
