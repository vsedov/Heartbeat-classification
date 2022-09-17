# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from heart.core import hc
from heart.utils.defaults.model import DefaultModel

keep_prob = 0.2


class CNNConv1d(DefaultModel):

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


class Conv1dV2(nn.Module):

    def __init__(self, input_features, output_dim):
        super().__init__()
        self.conv0 = nn.Conv1d(input_features, 128, output_dim, stride=1, padding=0)
        self.conv1 = nn.Conv1d(128, 128, output_dim, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(5, 2)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        inp = x.view(32, -1, 187)
        a_11 = F.relu(self.conv0(inp))
        c_22 = self.conv1(F.relu(self.conv1(self.pool1(torch.add(self.conv1(a_11), self.conv0(inp))))))
        m_21 = self.pool1(torch.add(c_22, self.pool1(torch.add(self.conv1(a_11), self.conv0(inp)))))
        m_31 = self.pool1(torch.add(self.conv1(F.relu(self.conv1(m_21))), m_21))
        m_41 = self.pool1(torch.add(self.conv1(F.relu(self.conv1(m_31))), m_31))
        m_51 = self.pool1(torch.add(self.conv1(F.relu(self.conv1(m_41))), m_41))
        return self.softmax(self.fc2(F.relu(self.fc1(m_51.view(32, -1)))))


def get_summary():
    summary(CNNConv1d(2, 5).to(hc.DEFAULT_DEVICE), (32, 187))
