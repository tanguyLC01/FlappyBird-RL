import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(12, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_space)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x