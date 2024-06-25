import numpy as np
import matplotlib.pyplot as plt

import imageio

from utils_3d import * 

def main():
    # Define the link parameters
    link_radius = 0.15
    link_length = 1.0
    num_links = 4

    # Define the initial edge lengths
    edge_lengths = np.ones((num_links, 2)) * link_length

    # Define the time step and number of steps for the simulation
    dt = 0.02
    num_steps = 200

    # Create a list to store the video frames
    frames = []

    # Generate random control signals
    control_signals = np.random.uniform(-0.15, 0.15, size=2 * num_links)

    for step in range(num_steps):

        # Generate a random perturbation vector
        perturbation = np.random.uniform(-0.06, 0.06, size=2 * num_links)
    
        # Update the control signal with the perturbation, Clip the control signal to the desired range
        control_signals = np.clip(control_signals + perturbation, -0.15, 0.15)

        # Integrate the control signals to update the edge lengths
        edge_lengths += control_signals.reshape(num_links, 2) * dt

        # Clip the edge lengths to ensure they stay within valid ranges
        edge_lengths = np.clip(edge_lengths, 0.8 * link_length, 1.2 * link_length)


        # Create a new figure and 3D axis for each frame
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Define the initial base center and normal
        base_center = np.zeros(3)
        base_normal = np.array([0, 0, 1])

        # Plot the links
        legend_elements, end_center, end_normal = plot_links_3d(edge_lengths, link_radius, link_length, ax, base_center, base_normal)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((4, 4, 4))

        # Add legend
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # Save the current frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        plt.close(fig)

    # Save the frames as a video using imageio
    imageio.mimsave('continuum_arm_simulation.mp4', frames, fps=20)

    print("Video saved as 'continuum_arm_simulation.mp4'")

if __name__ == '__main__':
    main()