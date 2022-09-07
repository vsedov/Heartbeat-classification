# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from heart.core import hc, hp

keep_prob = 0.2


class CNNConv1d(nn.Module):

    def __init__(self, input_features, output_dim):
        super().__init__()
        # 1-dimensional convolutional layer
        self.conv0 = nn.Conv1d(input_features, 128, output_dim, stride=1, padding=0)
        self.conv1 = nn.Conv1d(128, 128, output_dim, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(5, 2)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # I had to do this functionally, i wasnt sure how else to do it . without using F.relu
        x = x.view(32, -1, 187)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(32, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, keep_prob)
        x = self.fc2(x)
        return self.softmax(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = CNNConv1d(2, 5).to(device)

# check keras-like model summary using torchsummary
from torchsummary import summary

summary(model2, (32, 187))
