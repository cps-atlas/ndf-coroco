import torch.nn as nn

class CSDFNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_links, num_layers=4):
        super(CSDFNet, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.hidden_layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.hidden_layers.append(nn.Linear(hidden_size, num_links))
        self.softplus = nn.Softplus()

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.softplus(layer(x))
        x = self.hidden_layers[-1](x)
        return x