import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        x = self.leaky_relu(self.fc1(z))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
