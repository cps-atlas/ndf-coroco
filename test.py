import numpy as np
import matplotlib.pyplot as plt

# Define link parameters
link_nominal_length = 1.0  # Nominal length of the link
link_width = 0.3  # Width of the link

# Define the deformation
left_length_change = -0.1  # Change in left edge length

# Calculate the right edge length and curvature
left_length = link_nominal_length + left_length_change
right_length = 2 * link_nominal_length - left_length
curvature = (right_length - left_length) / (link_width * link_nominal_length)
print('curvature:', curvature)

# Calculate the radius and center of the deformed link
radius = (link_width / 2) * np.abs((left_length + right_length) / (right_length - left_length))
print('radius:', radius)
center_x = -radius
center_y = 0

# Calculate the central angle (theta) based on the arc length and radius
theta = link_nominal_length / radius

print('theta:', theta)

# Generate points for the original link
x_original = [-link_width/2, link_width/2, link_width/2, -link_width/2, -link_width/2]
y_original = [0, 0, link_nominal_length, link_nominal_length, 0]

# Generate points for the deformed link
theta_range = np.linspace(0, theta, 100)
x_left = center_x + (radius - link_width / 2) * np.cos(theta_range)
y_left = center_y + (radius - link_width / 2) * np.sin(theta_range)
x_right = center_x + (radius + link_width / 2) * np.cos(theta_range)
y_right = center_y + (radius + link_width / 2) * np.sin(theta_range)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the original link
ax.fill(x_original, y_original, 'b', alpha=0.3, edgecolor='b', linewidth=2, label='Original Link')

# Plot the deformed link
ax.fill(np.concatenate((x_left, x_right[::-1])), np.concatenate((y_left, y_right[::-1])), 'r', alpha=0.3, edgecolor='r', linewidth=2, label='Deformed Link')

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Link Deformation')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.show()