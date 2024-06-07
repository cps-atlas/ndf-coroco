import numpy as np
import matplotlib.pyplot as plt

def calculate_link(left_base, right_base, nominal_length, left_length):
    # Calculate link parameters
    link_width = np.sqrt((right_base[0] - left_base[0])**2 + (right_base[1] - left_base[1])**2)
    base_center = [(left_base[0] + right_base[0]) / 2, (left_base[1] + right_base[1]) / 2]

    # Calculate the right edge length and curvature
    right_length = 2 * nominal_length - left_length
    

    # Calculate the slope and angle of the line connecting the left and right bases
    base_angle = np.arctan2(right_base[1] - left_base[1], right_base[0] - left_base[0])

    num_of_edge_pts = 50


    # Check if the link is deformed
    if np.abs(right_length - left_length) < 1e-6:
        # Link is not deformed
        # Generate points for the original link
        theta_range = np.linspace(0, nominal_length, num_of_edge_pts)
        x_left = left_base[0] + theta_range * np.cos(base_angle+np.pi/2)
        y_left = left_base[1] + theta_range * np.sin(base_angle+np.pi/2)
        x_right = right_base[0] + theta_range * np.cos(base_angle+np.pi/2)
        y_right = right_base[1] + theta_range * np.sin(base_angle+np.pi/2)


        left_end = [x_left[-1], y_left[-1]]
        right_end = [x_right[-1], y_right[-1]]

        # Return the left edge points from top to bottom, right edge points from bottom to top
        return left_end, right_end, np.vstack((x_left[::-1], y_left[::-1])).T, np.vstack((x_right, y_right)).T
    else:
        # Calculate the radius and center of the deformed link
        radius = (link_width / 2) * np.abs((left_length + right_length) / (right_length - left_length))

        center_x = base_center[0] - np.sign(right_length - left_length) * radius * np.cos(base_angle)
        center_y = base_center[1] - np.sign(right_length - left_length) * radius * np.sin(base_angle)


        # Calculate the central angle (theta) based on the arc length and radius
        theta = nominal_length / radius

        # Calculate the starting angle of the arc
        start_angle = np.arctan2(base_center[1] - center_y, base_center[0] - center_x)


        # Generate points for the deformed link
        if left_length < right_length:
            theta_range = np.linspace(start_angle, start_angle + theta, num_of_edge_pts)

            x_left = center_x + (radius - link_width / 2) * np.cos(theta_range)
            y_left = center_y + (radius - link_width / 2) * np.sin(theta_range)
            x_right = center_x + (radius + link_width / 2) * np.cos(theta_range)
            y_right = center_y + (radius + link_width / 2) * np.sin(theta_range)
        else:
            theta_range = np.linspace(start_angle, start_angle - theta, num_of_edge_pts)

            x_left = center_x + (radius + link_width / 2) * np.cos(theta_range)
            y_left = center_y + (radius + link_width / 2) * np.sin(theta_range)
            x_right = center_x + (radius - link_width / 2) * np.cos(theta_range)
            y_right = center_y + (radius - link_width / 2) * np.sin(theta_range)



        left_end = [x_left[-1], y_left[-1]]
        right_end = [x_right[-1], y_right[-1]]


        # return the left edge points from top to bottom, right edge points from bottom to top;
        # this is important when checking if point is inside a polygon (require an ordered boundary points of a polygon)
        return left_end, right_end, np.vstack((x_left[::-1], y_left[::-1])).T, np.vstack((x_right, y_right)).T


def plot_links(link_lengths, nominal_length, left_base, right_base, ax):
    # Define a list of colors for each link
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    legend_elements = []

    # Iterate over the link lengths and plot each link
    for i, left_length in enumerate(link_lengths):
        # Calculate the current link
        left_end, right_end, left_coords, right_coords = calculate_link(left_base, right_base, nominal_length, left_length)


        # Concatenate the left and right edge coordinates
        x_coords = np.concatenate((left_coords[:, 0], right_coords[:, 0]))
        y_coords = np.concatenate((left_coords[:, 1], right_coords[:, 1]))

        # Plot the deformed link
        fill = ax.fill(x_coords, y_coords, colors[i % len(colors)], alpha=0.4, edgecolor=colors[i % len(colors)], linewidth=3, label=f'Link {i+1}')
        
        # Collect legend entry for the current link
        legend_elements.append(fill[0])

        # Update the base points for the next link
        left_base = left_end
        right_base = right_end

    ax.set_xlim(-2.5, 4)
    ax.set_ylim(0.0, len(link_lengths) * nominal_length + 0.5)
    # Increase font size for axis labels
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    
    # Increase font size for axis numbers
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Increase font size for legend
    ax.legend(fontsize=18)
    #ax.set_title('Multiple Link Deformation')

    ax.set_aspect('equal')

    return legend_elements


def main():
    # Define the sequence of link lengths
    link_lengths = [0.9, 0.9, 1.1, 1]

    #link_lengths = [1.0, 1.0, 1.0, 1.35]
    nominal_length = 1.0

    # Define the initial base points for the first link
    left_base = [-0.15, 0.0]
    right_base = [0.15, 0.0]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the links
    plot_links(link_lengths, nominal_length, left_base, right_base, ax)

    plt.show()

if __name__ == '__main__':
    main()