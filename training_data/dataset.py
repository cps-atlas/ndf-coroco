import torch
from torch.utils.data import Dataset

import jax.numpy as jnp

class SoftRobotDataset(Dataset):
    def __init__(self, configurations, points, distances):
        self.configurations = torch.from_numpy(configurations).float()
        self.points = torch.from_numpy(points).float()
        self.distances = torch.from_numpy(distances).float()
        #self.normals = torch.from_numpy(normals).float()

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx):
        inputs = torch.cat((self.configurations[idx], self.points[idx]))
        inputs.requires_grad = True
        #return inputs, self.distances[idx], self.normals[idx]
        return inputs, self.distances[idx]
    

class SoftRobotDataset_JAX:
    def __init__(self, configurations, points, distances):
        self.configurations = jnp.array(configurations, dtype = jnp.float32)
        self.points = jnp.array(points, dtype = jnp.float32)
        self.distances = jnp.array(distances, dtype = jnp.float32)

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx):
        config = self.configurations[idx]
        #print('config:', config.shape)

        point = self.points[idx]
        #print('pts:', point.shape)
        distance = self.distances[idx]
        inputs = jnp.concatenate((config, point), axis = 1)

        #print('inputs:', inputs.shape)

        return inputs, distance

