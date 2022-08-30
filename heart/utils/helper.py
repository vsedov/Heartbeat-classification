import os

import matplotlib.pyplot as plt
import torch
from torchinfo import summary

from heart.constants import DEFAULT_DEVICE


def to_default_device(*pointer):
    for output in pointer:
        yield output.to(DEFAULT_DEVICE)


def show_summary(network):
    summary(network)


def train_epoch(net, data_loader, loss_fn, optimizer=None, lr=0.01):

    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_loss, acc, count = 0, 0, 0
    for features, labels in data_loader:
        feat, lbls = to_default_device(features, labels)
        optimizer.zero_grad()
        out = net(feat)
        loss = loss_fn(out, lbls)  # cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == lbls).sum()
        count += len(labels)
    return total_loss.item() / count, acc.item() / count


def plot_results(hist):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()
