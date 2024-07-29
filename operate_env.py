

class Environment:
    def __init__(self, obstacle_positions, obstacle_velocities, obst_radius, goal_point, period=4):
        self.num_obstacles = len(obstacle_positions)
        self.obstacle_positions = obstacle_positions
        self.initial_velocities = obstacle_velocities.copy()  
        self.obstacle_velocities = obstacle_velocities
        self.obst_radius = obst_radius
        self.goal_point = goal_point
        self.period = period  # Total period of motion
        self.time = 0  # Keep track of elapsed time

    def update_obstacles(self, dt):
        self.time += dt
        t = self.time % self.period  # Time within the current period
        
        # Calculate the velocity based on the current time in the period
        if t < self.period / 2:
            # First half of the period: move in the initial direction
            self.obstacle_velocities = self.initial_velocities
        else:
            # Second half of the period: move in the opposite direction
            self.obstacle_velocities = -self.initial_velocities
        
        self.obstacle_positions += self.obstacle_velocities * dt
