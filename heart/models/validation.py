import torch

from heart.core import hc


def validate(net, dataloader, loss_fn):
    net.eval()
    count, acc, loss = 0, 0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            lbls = labels.to(hc.DEFAULT_DEVICE)
            out = net(features.to(hc.DEFAULT_DEVICE))
            loss += loss_fn(out, lbls)
            pred = torch.max(out, 1)[1]
            acc += (pred == lbls).sum()
            count += len(labels)
    return loss.item() / count, acc.item() / count
