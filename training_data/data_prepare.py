import numpy as np
import random
import sys
import os
import pickle

import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualize_soft_link import calculate_link, plot_links

import matplotlib.path as mpath

from shapely.geometry import Point, Polygon

def is_point_inside_link(point, boundary_points):
    # Create a Polygon object from the boundary points
    polygon = Polygon(boundary_points)
    point_to_check = Point(point)
    return polygon.contains(point_to_check)




def prepare_training_data(num_configs, length_limit, num_points, num_links, nominal_length, left_base, right_base, x_range, y_range):
    dataset = []

    for _ in range(num_configs):
        # Sample a random configuration
        left_lengths = np.random.uniform(nominal_length - length_limit, nominal_length + length_limit, num_links)

        # Generate boundary points for the current configuration
        boundary_points = []
        current_left_base = left_base
        current_right_base = right_base

        top_bot_edge_pt_num = 25

        # Append bottom edge points
        bottom_edge_points = np.linspace(current_left_base, current_right_base, top_bot_edge_pt_num)
        boundary_points.extend(bottom_edge_points.tolist())

        left_edge_points = []

        # Append right edge points (from link 1 to link N)
        for i in range(num_links):
            left_end, right_end, left_coords, right_coords = calculate_link(current_left_base, current_right_base, nominal_length, left_lengths[i])
            boundary_points.extend(right_coords.tolist())
            left_edge_points.append(left_coords.tolist())
            current_left_base = left_end
            current_right_base = right_end

        # Append top edge points (from right to left)
        top_edge_points = np.linspace(current_right_base, current_left_base, top_bot_edge_pt_num)
        boundary_points.extend(top_edge_points.tolist())

        # Append left edge points (from link N to link 1)
        for left_coords in reversed(left_edge_points):
            boundary_points.extend(left_coords)


        # Add boundary points to the dataset with distance 0
        for point in boundary_points:
            entry = {
                'configurations': left_lengths,
                'point': np.array(point),
                'distances': [0] * num_links
            }
            dataset.append(entry)

        for _ in range(num_points):
            # Sample a random workspace point
            point = np.array([random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])])

            distances = []
            current_left_base = left_base
            current_right_base = right_base
            for i in range(num_links):
                # Calculate the link positions
                left_end, right_end, left_coords, right_coords = calculate_link(current_left_base, current_right_base, nominal_length, left_lengths[i])

                # Find the closest point on the link to the workspace point
                link_points = np.vstack((left_coords, right_coords))
                distances_to_link = np.linalg.norm(link_points - point, axis=1)
                min_distance = np.min(distances_to_link)

                # Determine the sign of the distance
                inside_link = is_point_inside_link(point, boundary_points)

                if inside_link:
                    min_distance *= -1

                distances.append(min_distance)
                current_left_base = left_end
                current_right_base = right_end


            # Create a dataset entry
            entry = {
                'configurations': left_lengths,
                'point': point,
                'distances': distances
            }
            dataset.append(entry)

    return dataset


def visualize_dataset(dataset, num_links, nominal_length, left_base, right_base):
    # Select a random entry from the dataset
    entry = random.choice(dataset)
    left_lengths = entry['configurations']
    point = entry['point']
    distances = entry['distances']

    # Plot the soft robot configuration
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_links(left_lengths, nominal_length, left_base, right_base, ax)

    # Plot the boundary points for the selected configuration
    boundary_points = [entry['point'] for entry in dataset if np.all(np.array(entry['distances']) == 0) and np.array_equal(entry['configurations'], left_lengths)]
    boundary_points = np.array(boundary_points)
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], color='purple', label='Boundary Points', s=5)

    # Plot the sampled points with different colors based on the distance
    inside_points = [entry['point'] for entry in dataset if np.any(np.array(entry['distances']) < 0) and np.array_equal(entry['configurations'], left_lengths)]
    outside_points = [entry['point'] for entry in dataset if np.all(np.array(entry['distances']) > 0) and np.array_equal(entry['configurations'], left_lengths)]

    inside_points = np.array(inside_points)
    outside_points = np.array(outside_points)

    ax.scatter(inside_points[:, 0], inside_points[:, 1], color='red', label='Inside Points')
    ax.scatter(outside_points[:, 0], outside_points[:, 1], color='blue', label='Outside Points')

    # Plot the selected point
    ax.scatter(point[0], point[1], color='green', marker='x', s=100, label='Selected Point')

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Dataset Visualization')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Define the region of interest
    x_range = (-4, 4)
    y_range = (-1, 5)

    # Define the number of links and nominal length
    num_links = 4
    nominal_length = 1.0

    # Define the initial base points for the first link
    left_base = np.array([-0.15, 0.0])
    right_base = np.array([0.15, 0.0])

    # Prepare the training data
    num_configs = 200
    length_limit = 0.2

    num_points = 3000
    dataset = prepare_training_data(num_configs, length_limit, num_points, num_links, nominal_length, left_base, right_base, x_range, y_range)

    # Save the dataset to a file
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    # Visualize the prepared dataset
    visualize_dataset(dataset, num_links, nominal_length, left_base, right_base)