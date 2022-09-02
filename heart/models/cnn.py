# import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from heart.utils.defaults.model import DefaultModel

# import torch.nn as F

keep_prob = 0.2


class CNN(nn.Module):

    def __init__(self, class_val=5):
        super(CNN, self).__init__()
        #  OPTIM: (vsedov) (12:01:27 - 31/08/22): If you want to manually fine tune your parameters for each
        #  sequence, then i recommend you hard code this, else we can use a default param dict for  convenience .

        self.params = {
            'encoder': {
                'conv1': {
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                'conv2': {
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
            }
        }

        self.core_model = nn.Sequential(
            nn.Conv1d(1, 16, **self.params['encoder']['conv1']),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 64, **self.params['encoder']['conv1']),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, **self.params['encoder']['conv1']),
            nn.MaxPool1d(2),
        )

        self.linear = nn.Sequential(
            nn.Linear(2944, 500), nn.LeakyReLU(inplace=True), nn.Linear(500, class_val), nn.Softmax())

    def forward(self, x: DataLoader):
        x = self.core_model(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return self.linear(x)

