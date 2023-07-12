from abc import abstractmethod

import numpy as np
import torch.nn as nn


class DefaultModel(nn.Module):
    """
    Default class for majority of the models we will be using
    """

    @abstractmethod
    def forward(self, *inputs):
        raise NotADirectoryError

    def __str__(self):

        return (
            super().__str__()
            + f'\nParameters_Trained: {sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters()))}'
        )
