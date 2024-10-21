import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, dtype=torch.float64)
        self.fc2 = nn.Linear(10, 10, dtype=torch.float64)
        self.fc3 = nn.Linear(10, 1, dtype=torch.float64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def print_gradient(self):
        pass

    
def set_model_weights(data: dict, model: nn.Module):

    model.fc1.weight = nn.Parameter(torch.tensor(data['w1'], dtype=torch.float64))
    model.fc1.bias = nn.Parameter(torch.tensor(data['b1'], dtype=torch.float64))
    model.fc2.weight = nn.Parameter(torch.tensor(data['w2'], dtype=torch.float64))
    model.fc2.bias = nn.Parameter(torch.tensor(data['b2'], dtype=torch.float64))
    model.fc3.weight = nn.Parameter(torch.tensor(data['w3'], dtype=torch.float64))
    model.fc3.bias = nn.Parameter(torch.tensor(data['b3'], dtype=torch.float64))
    return 