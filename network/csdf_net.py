import torch.nn as nn

from flax import linen as jnn
from jax.nn.initializers import he_uniform

class CSDFNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(CSDFNet, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.hidden_layers.append(nn.Linear(hidden_size, output_size))
        self.softplus = nn.Softplus()

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.softplus(layer(x))
        x = self.hidden_layers[-1](x)
        return x
    

class CSDFNet_JAX(jnn.Module):
    hidden_size: int
    output_size: int
    num_layers: int = 4

    @jnn.compact
    def __call__(self, x):
        for _ in range(self.num_layers - 1):
            x = jnn.Dense(self.hidden_size, kernel_init=he_uniform())(x)
            x = jnn.softplus(x)
        x = jnn.Dense(self.output_size, kernel_init=he_uniform())(x)
        return x
