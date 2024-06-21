import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils import plot_links


def is_valid_node(node, obstacles, obst_radius, grid_resolution):
    node_pos = np.array(node) * grid_resolution
    for obstacle in obstacles:
        if np.linalg.norm(node_pos - obstacle) <= obst_radius:
            return False
    return True

def heuristic(node, goal):
    return np.linalg.norm(np.array(node) - np.array(goal))

def astar(grid, start, goal, obstacles, obst_radius, grid_resolution):
    rows, cols = grid.shape
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = min(open_set, key=lambda node: f_score[node])

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        open_set.remove(current)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0 and is_valid_node(neighbor, obstacles, obst_radius, grid_resolution):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

    return None

def main():
    # Define the robot parameters
    num_links = 4
    nominal_length = 1.0
    link_lengths = np.ones(num_links) * nominal_length
    left_base = np.array([-0.15, 0.0])
    right_base = np.array([0.15, 0.0])

    # Define the obstacle positions and radius
    obstacle_positions = np.array([ [2.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
    obst_radius = 0.3

    # Define the goal position
    goal_position = np.array([2.8, 1.5])

    # Create a grid for A* path planning
    grid_size = (50, 50)
    grid_resolution = 0.1

    # Create a grid representation of the environment
    grid = np.zeros(grid_size, dtype=int)

    # Mark obstacles in the grid
    for obstacle in obstacle_positions:
        obst_node = tuple(np.round(obstacle / grid_resolution).astype(int))
        grid[obst_node] = 1

    # Define the start and goal nodes
    start_node = tuple(np.round(np.array([0.0, 4.0]) / grid_resolution).astype(int))
    goal_node = tuple(np.round(np.array([2.8, 1.5])/ grid_resolution).astype(int))

    # Update the obstacle marking in the grid
    for obstacle in obstacle_positions:
        obst_node = tuple(np.round(obstacle / grid_resolution).astype(int))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                node = (i, j)
                if not is_valid_node(node, [obstacle], obst_radius, grid_resolution):
                    grid[node] = 1

    # Run the A* algorithm
    path = astar(grid, start_node, goal_node, obstacle_positions, obst_radius+0.1, grid_resolution)


    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the links using the plot_links function
    legend_elements = plot_links(link_lengths, nominal_length, left_base, right_base, ax)

    # Plot the goal position
    goal_plot = ax.plot(goal_position[0], goal_position[1], marker='*', markersize=15, color='blue', label='Goal')[0]
    legend_elements.append(goal_plot)

    # Plot the obstacles
    for obstacle in obstacle_positions:
        obstacle_circle = Circle(obstacle, radius=obst_radius, color='r', fill=True)
        ax.add_patch(obstacle_circle)
    obstacle_plot = ax.plot(obstacle_positions[0, 0], obstacle_positions[0, 1], 'ro', markersize=10, label='Obstacles')[0]
    legend_elements.append(obstacle_plot)

    # Plot the path
    if path is not None:
        path_points = np.array(path) * grid_resolution
        ax.plot(path_points[:, 0], path_points[:, 1], 'g--', label='A* Path')
    else:
        print("No path found.")

    # Set the plot limits and labels
    # ax.set_xlim(-1, 4)
    # ax.set_ylim(-1, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('A* Path Planning')
    ax.legend(handles=legend_elements)

    plt.savefig('a_star_path_planning.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()