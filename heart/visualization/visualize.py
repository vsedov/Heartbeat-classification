import os

import matplotlib.pyplot as plt
import torch.nn as nn

from heart.core import hc
from heart.log import get_logger

log = get_logger(__name__)


def visualise_network(net: nn.Sequential):
    weight_input = next(net.parameters())
    fig, ax = plt.subplot(1, 10, figsize=(15, 4))
    for i, x in enumerate(weight_input):
        ax[i].imshow(x.view(28, 28).detach())
    ax.show()


def show_data(name, model_type, hist):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()

    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()
    file_path = f"{hc.DIR}reports/figures/{model_type}/"
    filename = f"{file_path}{model_type}-{name}"

    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))


def show_validation_loss(hist):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(hist['test_loss'], label='Test acc')
    plt.plot(hist['test_acc'], label='Test acc')
    plt.legend()
    plt.show()


def visualize_data_loader(data_loader):
    fig, ax = plt.subplots(1, 10, figsize=(15, 4))
    for i, (x, y) in enumerate(data_loader):
        ax[i].imshow(x.view(28, 28).detach())
    plt.show()
