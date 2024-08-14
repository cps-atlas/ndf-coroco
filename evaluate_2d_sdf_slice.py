import numpy as np
import matplotlib.pyplot as plt

from robot_config import *
import jax.numpy as jnp
from jax import jit
from utils_3d import *

import mcubes
import imageio



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

def main():
    fig = plt.figure(figsize=(12, 10), dpi=150)
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
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
               s=10, c='black', alpha=1, label='Robot Surface')

    # Plot the 3D slice level set
    plot_3d_slice_level_set(surface_points, ax)



    ax.set_axis_off()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 3)

    plt.tight_layout()
    plt.savefig("extensible_link_continuum_with_slice_level_set.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

'''
following is for creating the video
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

    
