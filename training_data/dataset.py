import torch
from torch.utils.data import Dataset

class SoftRobotDataset(Dataset):
    def __init__(self, configurations, points, distances):
        self.configurations = torch.from_numpy(configurations).float()
        self.points = torch.from_numpy(points).float()
        self.distances = torch.from_numpy(distances).float()

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx):
        inputs = torch.cat((self.configurations[idx], self.points[idx]))
        inputs.requires_grad = True  
        return inputs, self.distances[idx]
    

