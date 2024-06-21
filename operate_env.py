import numpy as np

from utils import compute_cbf_value_and_grad

class Environment:
    def __init__(self, obstacle_positions, obstacle_velocities, obst_radius, goal_point):
        self.num_obstacles = len(obstacle_positions)
        self.obstacle_positions = obstacle_positions
        self.obstacle_velocities = obstacle_velocities
        self.obst_radius = obst_radius
        self.goal_point = goal_point

    def update_obstacles(self, dt):
        self.obstacle_positions += self.obstacle_velocities * dt

    # def check_collision(self, robot, obst_radius):
    #     for link_length in robot.link_lengths:
    #         cbf_h_val, _, _ = compute_cbf_value_and_grad(jax_params, np.array([link_length]), self.obstacle_positions, self.obstacle_velocities)
    #         if np.min(cbf_h_val) - obst_radius < 0:
    #             return True
    #     return False