import torch
import torch.nn as nn
import torch.nn.functional as F


class FNNet(nn.Module):

    def __init__(self, input_dim, interm_dim, output_dim):
        super(FNNet, self).__init__()

        self.input_dim = input_dim
        self.dp1 = torch.nn.Dropout(0.2)
        self.dp2 = torch.nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, interm_dim)
        self.fc2 = nn.Linear(interm_dim, interm_dim)
        self.fc3 = nn.Linear(interm_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc3(x)
        return x

    def embed(self, x):
        x = self.dp1(F.relu(self.fc1(x.view(-1, self.input_dim))))
        x = self.dp2(F.relu(self.fc2(x)))
        return x


class ConvNet(nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.dp3 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc2(x)
        return x

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.dp3(self.fc1(x)))
        return x
