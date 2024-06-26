import jax.numpy as jnp
from utils import get_last_link_grasp_points


class Robot:
    def __init__(self, num_links, nominal_length, left_base, right_base):
        self.num_links = num_links
        self.nominal_length = nominal_length
        self.left_base = left_base
        self.right_base = right_base
        self.link_lengths = jnp.ones(num_links) * nominal_length

    def update_link_lengths(self, control_signals, dt):

        # forward euler
        # self.link_lengths += control_signals * dt

        # RK45
        k1 = control_signals
        k2 = control_signals + 0.5 * dt * k1
        k3 = control_signals + 0.5 * dt * k2
        k4 = control_signals + dt * k3
        self.link_lengths += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_last_link_grasp_points(self):
        return get_last_link_grasp_points(self.left_base, self.right_base, self.link_lengths, self.nominal_length)
    


# class Robot:
#     def __init__(self, jax_params, num_links, nominal_length, left_base, right_base):
#         self.jax_params = jax_params
#         self.num_links = num_links
#         self.nominal_length = nominal_length
#         self.left_base = left_base
#         self.right_base = right_base
#         self.link_lengths = jnp.ones(num_links) * nominal_length

#     def update_link_lengths(self, control_signals, dt):
#         self.link_lengths = integrate_link_lengths(self.link_lengths, control_signals, dt)

#     def get_last_link_grasp_points(self):
#         return get_last_link_grasp_points(self.left_base, self.right_base, self.link_lengths, self.nominal_length)