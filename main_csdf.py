import pickle
import numpy as np
import torch
from utils.csdf_net import CSDFNet
from training.csdf_training import train
from training.config import *

from training_data.dataset import SoftRobotDataset

from torch.utils.data import DataLoader

def main():
    # Load the saved dataset from the pickle file
    with open('training_data/dataset.pickle', 'rb') as f:
        training_data = pickle.load(f)

    # Extract configurations, workspace points, and distances from the dataset
    configurations = np.array([entry['configurations'] for entry in training_data])
    points = np.array([entry['point'] for entry in training_data])
    distances = np.array([entry['distances'] for entry in training_data])

    # Create dataset and data loaders
    dataset = SoftRobotDataset(configurations, points, distances)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE)

    # Create neural network
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    # Train the model
    net = train(net, train_dataloader, val_dataloader, NUM_EPOCHS, LEARNING_RATE, device=device, loss_threshold=0.005, lambda_eikonal=0.1)

    # Save the trained model
    torch.save(net.state_dict(), "trained_model_no_eikonal.pth")

if __name__ == "__main__":
    main()