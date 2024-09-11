import numpy as np
import matplotlib.pyplot as plt

from robot_config import *
import jax.numpy as jnp
from jax import jit
from utils_3d import *

import mcubes
import imageio

from evaluate_heatmap import load_learned_csdf
from utils_3d import *



def generate_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z

@jit
def compute_distance(points, surface_points):
    distances = jnp.min(jnp.linalg.norm(points[:, jnp.newaxis, :] - surface_points[jnp.newaxis, :, :], axis=2), axis=1)
    return distances

def plot_distance_field_and_isosurfaces(surface_points, ax, max_distance=0.5, level_set_values=[0.2, 0.3, 0.4]):
    x_min, x_max, y_min, y_max, z_min, z_max = -2, 2, -2, 2, -1, 3
    resolution = 50  # Increased for better isosurface resolution
    X, Y, Z = generate_grid_points(x_min, x_max, y_min, y_max, z_min, z_max, resolution)
    
    # Reshape grid_points for vectorized computation
    grid_points_flat = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    
    # Compute distances using JAX
    distances = compute_distance(jnp.array(grid_points_flat), jnp.array(surface_points))
    distances = np.array(distances.reshape(X.shape))  # Convert to NumPy array
    
    # Plot scatter points
    mask = distances.flatten() <= max_distance
    plot_points = grid_points_flat[mask]
    plot_distances = distances.flatten()[mask]
    # scatter = ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2],
    #                      c=plot_distances, cmap='viridis', s=5, alpha=0.3)
    
    # Plot isosurfaces
    colors = plt.cm.viridis(np.linspace(0, 1, len(level_set_values)))
    for value, color in zip(level_set_values, colors):
        verts, faces = mcubes.marching_cubes(distances, value)
        verts = verts / resolution
        verts = verts * [x_max-x_min, y_max-y_min, z_max-z_min] + [x_min, y_min, z_min]
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=color, alpha=0.3)

# this plots the ground truth level set
def plot_3d_slice_level_set(surface_points, ax, x_slice=0, resolution=100, level_set_values=[0.25, 0.3, 0.35, 0.4, 0.5]):
    y_min, y_max, z_min, z_max = -2, 2, -1, 3
    
    # Generate 2D grid for the slice
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    Y, Z = np.meshgrid(y, z)
    
    # Create 3D points for the slice
    slice_points = np.column_stack((np.full(Y.size, x_slice), Y.flatten(), Z.flatten()))
    
    # Compute distances using JAX
    distances = compute_distance(jnp.array(slice_points), jnp.array(surface_points))
    distances = np.array(distances.reshape(Y.shape))  # Convert to NumPy array
    
    
    # Create a scatter plot of the points, colored by their distance values
    # scatter = ax.scatter(slice_points[:, 0], slice_points[:, 1], slice_points[:, 2],
    #                      c=distances.flatten(), cmap='viridis', s=1, alpha=0.3)
    

    
    # Plot 3D contours
    contours = ax.contour3D(Y, Z, distances, levels=level_set_values, cmap='coolwarm')
    
    # Transform contour coordinates
    for collection in contours.collections:
        paths = collection.get_paths()
        for path in paths:
            vertices = path.vertices
            x = np.full_like(vertices[:, 0], x_slice)
            y = vertices[:, 0]
            z = vertices[:, 1]
            ax.plot(x, y, z, color=collection.get_edgecolor()[0], linewidth=2)
    
    # Remove the original contours (which are on the wrong plane)
    for collection in contours.collections:
        collection.remove()

@jit
def evaluate_single_link(jax_params, cable_lengths, points):
    # Simplified version for a single link
    rbt_configs = state_to_config(cable_lengths)
    inputs = jnp.hstack((jnp.repeat(rbt_configs.reshape(1, -1), points.shape[0], axis=0), points))
    outputs = CSDFNet_JAX(HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS).apply(jax_params, inputs)
    return outputs[:, 0]

# this plots the learned heatmap
def plot_3d_learned_slice_heatmap(net, cable_lengths, ax, x_slice=0, resolution=100, level_set_values=[0.2, 0.25, 0.3, 0.35, 0.4]):
    y_min, y_max, z_min, z_max = -1.5, 1.5, -1, 3
    #y_min, y_max, z_min, z_max = -2, 1, -1, 3
    
    # Generate 2D grid for the slice
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    Y, Z = np.meshgrid(y, z)
    
    # Create 3D points for the slice
    slice_points = np.column_stack((np.full(Y.size, x_slice), Y.flatten(), Z.flatten()))
    
    # Compute learned distances using JAX
    learned_distances = evaluate_single_link(net.params, cable_lengths[0], jnp.array(slice_points))
    learned_distances = np.array(learned_distances.reshape(Y.shape))  # Convert to NumPy array
    
    # Create a plane at x_slice
    X = np.full_like(Y, x_slice)
    
    # Plot heatmap using plot_surface
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(learned_distances), 
                           rstride=1, cstride=1, alpha=0.7, shade=False)
    
    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m.set_array(learned_distances)
    
    

    return ax

# this plots the learned level set
def plot_3d_learned_slice_level_set(net, cable_lengths, ax, x_slice=0, resolution=100, level_set_values=[0.2, 0.25, 0.3, 0.35, 0.4]):
    # Generate 2D grid for the slice
    y, z = np.linspace(-2, 2, resolution), np.linspace(-1, 3, resolution)
    
    Y, Z = np.meshgrid(y, z)
    
    # Create 3D points for the slice and evaluate learned distances
    slice_points = np.column_stack((np.full(Y.size, x_slice), Y.flatten(), Z.flatten()))
    learned_distances = evaluate_single_link(net.params, cable_lengths[0], jnp.array(slice_points))
    learned_distances = learned_distances.reshape(Y.shape)
    
    # Create and plot 3D contours
    contours = ax.contour3D(Y, Z, learned_distances, levels=level_set_values, cmap='coolwarm')
    
    # Transform contour coordinates to 3D space
    for collection in contours.collections:
        for path in collection.get_paths():
            vertices = path.vertices
            ax.plot(np.full_like(vertices[:, 0], x_slice), vertices[:, 0], vertices[:, 1], 
                    color=collection.get_edgecolor()[0], linewidth=2)
        collection.remove()

def plot_disk(center, radius, normal, ax, color='b', alpha=0.7):
    # Create points on a disk
    theta = np.linspace(0, 2*np.pi, 100)
    u = np.linspace(0, radius, 10)
    Theta, U = np.meshgrid(theta, u)
    
    X = U * np.cos(Theta)
    Y = U * np.sin(Theta)
    Z = np.zeros_like(X)

    # Rotate the disk to align with the given normal vector
    rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), normal)
    points = np.dot(rotation_matrix, np.array([X.flatten(), Y.flatten(), Z.flatten()]))
    
    # Reshape and translate the points
    X = points[0, :].reshape(X.shape) + center[0]
    Y = points[1, :].reshape(Y.shape) + center[1]
    Z = points[2, :].reshape(Z.shape) + center[2]

    # Plot the disk
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

def plot_link_surface(state, link_radius, ax, base_center=np.zeros(3), base_normal=np.array([0, 0, 1])):
    edge_lengths = compute_3rd_edge_length(state[0])
    edge_points = np.array(compute_edge_points(edge_lengths))  # Convert to numpy array


    # Apply rotation and translation if needed
    if not np.array_equal(base_center, np.array([0, 0, 0])):
        rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), base_normal)
        for j in range(len(edge_points)):
            rotated_edge_points = np.dot(rotation_matrix, edge_points[j])
            translated_edge_points = rotated_edge_points + base_center.reshape(3, 1)
            edge_points[j] = translated_edge_points

    # Calculate the end circle center, normal, and radius
    end_center, end_normal, end_radius = calculate_link_circle(edge_points)

    # Plot base and end circles
    plot_disk(base_center, link_radius, base_normal, ax, color='r')
    plot_disk(end_center, end_radius, end_normal, ax, color='r')

    # Create a mesh for the surface
    theta = np.linspace(0, 2*np.pi, 50)
    t = np.linspace(0, 1, edge_points.shape[2])
    Theta, T = np.meshgrid(theta, t)

    # Initialize arrays for the surface
    X = np.zeros_like(Theta)
    Y = np.zeros_like(Theta)
    Z = np.zeros_like(Theta)

    # Create the surface by interpolating circles at each level
    for i in range(edge_points.shape[2]):
        # Get the three edge points at this level
        points = edge_points[:, :, i]
        
        # Calculate the center and radius of the circle at this level
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        
        # Calculate the normal of the circle at this level
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Create a local coordinate system
        u = v1 / np.linalg.norm(v1)
        v = np.cross(normal, u)
        
        # Create a circle at this level
        circle_x = center[0] + radius * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
        circle_y = center[1] + radius * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
        circle_z = center[2] + radius * (np.cos(theta) * u[2] + np.sin(theta) * v[2])
        
        # Add to the surface arrays
        X[i, :] = circle_x
        Y[i, :] = circle_y
        Z[i, :] = circle_z

    # Plot the surface
    ax.plot_surface(X, Y, Z, color='r', alpha=1.0)


    return ax




def main():
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Define the link parameters
    link_radius = LINK_RADIUS
    link_length = LINK_LENGTH

    # Define the state for a single link
    state = jnp.array([[2.0, 2.0]])  # Adjust as needed for bending

    # Compute the surface points of the robot shape
    surface_points = compute_surface_points(state, link_radius, link_length, num_points_per_circle=70)
    surface_points = np.concatenate(surface_points, axis=0)

    # Plot the surface points of the robot
    # ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
    #            s=10, c='black', alpha=1, label='Robot Surface')
    
    plot_link_surface(state, link_radius, ax)

    # Plot the 3D slice level set
    # plot_3d_slice_level_set(surface_points, ax)

    model_type = 'jax'

    # trained_model = "trained_models/torch_models_3d/eikonal_train_4_16.pth"

    # paper table prepare
    trained_model = "trained_models/torch_models_3d/grid_search_moe_4_16.pth"

    net = load_learned_csdf(model_type, trained_model_path = trained_model)



    # plot_3d_learned_slice_level_set(net, state, ax, x_slice=0, resolution=100)
    plot_3d_learned_slice_heatmap(net, state, ax, x_slice=0, resolution=100)



    ax.set_axis_off()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 3)

    plt.tight_layout()
    plt.savefig("link_continuum_with_slice_level_set.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

'''
following can be used for creating the video
'''


# def create_frame(state, link_radius, link_length, ax):
#     ax.clear()
    
#     # Compute the surface points of the robot shape
#     surface_points = compute_surface_points(state, link_radius, link_length, num_points_per_circle=70)
#     surface_points = np.concatenate(surface_points, axis=0)
    

#     # Plot the surface points of the robot
#     ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
#                s=10, c='black', alpha=1, label='Robot Surface')
    
#     # Plot the distance field and isosurfaces
#     plot_distance_field_and_isosurfaces(surface_points, ax)
    
#     ax.set_axis_off()
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_zlim(-1, 3)

# def update_state(state, control_input, dt):
#     return state + control_input * dt

# def main():
#     # Define the link parameters
#     link_radius = LINK_RADIUS
#     link_length = LINK_LENGTH
    
#     # Set up the figure
#     fig = plt.figure(figsize=(12, 10), dpi=150)
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Define the number of frames and the range of states
#     num_frames = 100
#     state_range = (1.6, 2.4)
    
#     # Generate a sequence of states
#     states = np.linspace(state_range[0], state_range[1], num_frames)

    
#     # Create a list to store the video frames
#     frames = []
    
#     for i in range(num_frames):
#         state = jnp.array([[states[i], states[num_frames-1-i]]])  # Create varying states
        
#         create_frame(state, link_radius, link_length, ax)
        
#         # Convert the plot to an image
#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
#         frames.append(image)
        
#         print(f"Processed frame {i+1}/{num_frames}")
    
#     # Save the frames as a video
#     imageio.mimsave('robot_state_animation.mp4', frames, fps=30)
    
#     print("Video saved as 'robot_state_animation.mp4'")

# if __name__ == '__main__':
#     main()

    
