import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.csdf_net import CSDFNet
from training.config import *

from visualize_soft_link import plot_links

def evaluate_csdf(net, configuration, x_range, y_range, resolution=2):
    # Generate a grid of points in the workspace
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    points = np.stack((xx.flatten(), yy.flatten()), axis=1)

    # Prepare the input tensor
    configurations = np.repeat(configuration.reshape(1, -1), points.shape[0], axis=0)
    inputs = np.hstack((configurations, points))
    inputs_tensor = torch.from_numpy(inputs).float()

    # Evaluate the signed distance values
    net.eval()
    with torch.no_grad():
        outputs = net(inputs_tensor)
    min_sdf_distance = np.min(outputs.numpy(), axis=1)
    distances = min_sdf_distance.reshape(resolution, resolution)
    return distances

def plot_csdf_heatmap(configuration, distances, x_range, y_range, nominal_length, left_base, right_base, save_path=None):
    # Plot the soft robot configuration
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_links(configuration, nominal_length, left_base, right_base, ax)

    # Plot the signed distance field heatmap
    im = ax.imshow(distances, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], cmap='coolwarm', alpha=0.7)
    fig.colorbar(im, ax=ax, label='Signed Distance')

    # Plot the zero level set
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], distances.shape[0]), 
                         np.linspace(y_range[0], y_range[1], distances.shape[1]))
    contour = ax.contour(xx, yy, distances, levels=[0], colors='k', linewidths=2)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('C-SDF Heatmap')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save the figure in high resolution
    plt.show()

def main():
    # Load the trained model
    net = CSDFNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LINKS, NUM_LAYERS)
    net.load_state_dict(torch.load("baseline_3.pth"))
    net.eval()

    # Define the region of interest
    x_range = (-4, 4)
    y_range = (-1, 5)

    # Define the nominal length and initial base points
    nominal_length = 1.0
    left_base = np.array([-0.15, 0.0])
    right_base = np.array([0.15, 0.0])

    # Define the configuration to evaluate
    configuration = np.array([1.1, 1.1, 1.0, 1.05])

    # Evaluate the signed distance values
    distances = evaluate_csdf(net, configuration, x_range, y_range, resolution=200)

    # Plot the C-SDF heatmap
    plot_csdf_heatmap(configuration, distances, x_range, y_range, nominal_length, left_base, right_base, save_path='csdf_heatmap.png')

if __name__ == "__main__":
    main()