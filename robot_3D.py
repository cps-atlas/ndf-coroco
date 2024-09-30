import jax.numpy as jnp
from utils_3d import *


class Robot3D:
    def __init__(self, num_links, link_radius, link_length, base_center=jnp.zeros(3), base_normal=jnp.array([0, 0, 1])):
        self.num_links = num_links
        self.link_radius = link_radius
        self.link_length = link_length
        self.base_center = base_center
        self.base_normal = base_normal
        self.state = jnp.ones((num_links, 2)) * link_length

    def update_edge_lengths(self, control_signals, dt):
        # Reshape the control signals into the correct shape
        control_signals_reshaped = control_signals.reshape(self.num_links, 2)
        # Forward Euler integration
        self.state += control_signals_reshaped * dt
        # Clip the edge lengths to ensure they stay within valid ranges
        # self.edge_lengths = jnp.clip(self.edge_lengths, 0.8 * self.link_length, 1.2 * self.link_length)
