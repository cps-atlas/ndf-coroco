import numpy as np
import matplotlib.pyplot as plt


def calculate_link_parameters(edge_lengths, link_radius):
    q1, q2, q3 = edge_lengths
    r = link_radius

    # Calculate theta (bending angle)
    theta = 2 * np.sqrt(q1**2 + q2**2 + q3**2 - q1*q2 - q2*q3 - q1*q3) / (3 * r)

    # Calculate phi (bending direction angle)
    phi = np.arctan2(np.sqrt(3) * (q2 - q3), q2 + q3 - 2*q1)

    return theta, phi

def compute_edge_points(edge_lengths, link_radius, link_length):
    theta, phi = calculate_link_parameters(edge_lengths, link_radius)

    # Calculate the bending radius
    if abs(theta) < 1e-4:
        R = float('inf')
    else:
        R = link_length / theta

    # Generate points for the deformed link
    t = np.linspace(0, theta, 50)

    # Calculate points for each edge
    edge_points = []
    for i in range(3):
        # Calculate the angle for each cable
        cable_angle = i * 2 * np.pi / 3
        r = link_radius
        # Calculate the x, y, z coordinates for each point along the edge
        if R == float('inf'):
            t = np.linspace(0, link_length, 100)
            x = link_radius * np.cos(cable_angle) * np.ones_like(t)
            y = link_radius * np.sin(cable_angle) * np.ones_like(t)
            z =  t 
        else:
            x = R - (R - r * np.cos(cable_angle)) * np.cos(t)
            y = r * np.sin(cable_angle) * np.ones_like(t)
            z = (R - r * np.cos(cable_angle)) * np.sin(t)
        # Rotate the points based on phi
        rotated_x = x * np.cos(phi) - y * np.sin(phi)
        rotated_y = x * np.sin(phi) + y * np.cos(phi)
        edge_points.append(np.vstack((rotated_x, rotated_y, z)))

    return edge_points

def calculate_end_circle(edge_points):
    # Get the three points that construct the end circle
    p1 = edge_points[0][:, -1]
    p2 = edge_points[1][:, -1]
    p3 = edge_points[2][:, -1]

    # Compute the center of the end circle
    end_center = np.mean([p1, p2, p3], axis=0)

    # Compute the normal vector of the end circle
    v1 = p2 - p1
    v2 = p3 - p1
    end_normal = np.cross(v1, v2)
    end_normal = end_normal / np.linalg.norm(end_normal)

    radius = np.linalg.norm(end_center - p1)

    #print('check_radius:', radius)

    return end_center, end_normal, radius

def calculate_rotation_matrix(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)

    if s < 1e-6:
        return np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        return np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)
    
def plot_circle(center, radius, normal, ax, color='k'):
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)

    # Rotate the circle points based on the normal vector
    rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), normal)
    rotated_points = np.dot(rotation_matrix, np.vstack((x, y, z)))

    # Translate the rotated points based on the center
    translated_points = rotated_points + center.reshape(3, 1)

    ax.plot(translated_points[0], translated_points[1], translated_points[2], color=color, linewidth=2)


def plot_links_3d(states, link_radius, link_length, ax, base_center=np.zeros(3), base_normal=np.array([0, 0, 1])):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    legend_elements = []

    for i, state in enumerate(states):
        q1, q2 = state
        q3 = 3 * link_length - q1 - q2
        edge_lengths = [q1, q2, q3]

        # Plot the base circle
        plot_circle(base_center, link_radius, base_normal, ax)

        # Compute the deformed link points for each edge
        edge_points = compute_edge_points(edge_lengths, link_radius, link_length)

        # Apply rotation and translation to the edge points
        if not np.array_equal(base_center, np.array([0, 0, 0])):
            rotation_matrix = calculate_rotation_matrix(np.array([0, 0, 1]), base_normal)
            for j in range(len(edge_points)):
                rotated_edge_points = np.dot(rotation_matrix, edge_points[j])
                translated_edge_points = rotated_edge_points + base_center.reshape(3, 1)
                edge_points[j] = translated_edge_points

        # Plot the deformed edges
        for points in edge_points:
            ax.plot(points[0], points[1], points[2], color=colors[i], linewidth=2)

        # Calculate the end circle center, normal, and radius
        end_center, end_normal, end_radius = calculate_end_circle(edge_points)

        # print('end_center:', end_center)
        # print('end_mormal:', end_normal)

        # Plot the end circle if it's the last link
        if i == len(states) - 1:
            plot_circle(end_center, end_radius, end_normal, ax)

        # Collect legend entry for the current link
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Link {i+1}'))

        # Update the base center and normal for the next link
        base_center = end_center
        base_normal = end_normal

    return legend_elements, end_center, end_normal

def main():
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Define the link parameters
    link_radius = 0.15
    link_length = 1.0

    # Define the states for multiple links
    states = np.array([[1.0, 1.0], [1.2, 0.9], [0.9, 1.0], [1.1, 0.9]])

    # Define the initial base center and normal
    base_center = np.zeros(3)
    base_normal = np.array([0, 0, 1])

    # Plot the links
    legend_elements, end_center, end_normal = plot_links_3d(states, link_radius, link_length, ax, base_center, base_normal)

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
    plt.savefig("3d_continuum_arm_multiple_links.png", dpi=300)
    plt.show()

    print("End effector position:", end_center)

if __name__ == '__main__':
    main()