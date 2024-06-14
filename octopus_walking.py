import numpy as np
import matplotlib.pyplot as plt
import imageio
from visualize_soft_link import plot_links


def generate_gait_pattern(t, num_legs, num_links_per_leg, nominal_length, amplitude, phase_diff):
    link_lengths = []
    for leg in range(num_legs):
        phase_offset = leg * (2 * np.pi / num_legs) / 8
        for link in range(num_links_per_leg):
            link_length = nominal_length + amplitude * np.sin(t + phase_offset + link * phase_diff)
            link_lengths.append(link_length)
    return link_lengths

def main():
    # Define the number of legs and links per leg for the octopus-like robot
    num_legs = 2
    num_links_per_leg = 4

    # Define the nominal length, amplitude, and phase difference for the gait pattern
    nominal_length = 1.0
    amplitude = 0.15
    phase_diff = np.pi / 4

    # Define the initial base points for the first link of each leg
    base_points = [
        [[0.85, 0.0], [0.55, 0.0]],
        [[-0.55, 0.0], [-0.85, 0.0]]
    ]

    # Define the number of time steps and the time step size
    num_steps = 100
    dt = 0.1

    # Create a video writer object
    writer = imageio.get_writer("octopus_walking_animation.mp4", fps=int(1/dt))

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Simulate the octopus-like robot walking for the given time steps
    for step in range(num_steps):
        t = step * dt

        # Generate the link lengths based on the gait pattern
        link_lengths = generate_gait_pattern(t, num_legs, num_links_per_leg, nominal_length, amplitude, phase_diff)

        # Clear the previous plot
        ax.clear()

        # Plot the links for each leg using the plot_links function
        for leg in range(num_legs):
            start_index = leg * num_links_per_leg
            end_index = start_index + num_links_per_leg
            leg_link_lengths = link_lengths[start_index:end_index]
            left_base, right_base = base_points[leg]
            plot_links(leg_link_lengths, nominal_length, left_base, right_base, ax)

        # Set the plot limits and labels
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-1, 7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Convert the plot to an image and append it to the video
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        writer.append_data(image[:, :, :3])

    writer.close()

if __name__ == '__main__':
    main()