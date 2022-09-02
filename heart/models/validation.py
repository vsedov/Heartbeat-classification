import sys

import torch

from heart.core import hc, hp


def validate(net, dataloader, loss_fn):
    net.eval()
    acc, loss = 0, 0

    count = len(dataloader.dataset)
    with torch.no_grad():
        for features, labels in dataloader:
            lbls = labels.to(hc.DEFAULT_DEVICE)
            out = net(features.to(hc.DEFAULT_DEVICE))
            loss += loss_fn(out, lbls)
            # pred = torch.max(out, 1)[1]
            # REVISIT: (vsedov) (14:37:40 - 02/09/22): Something might be wrong here
            # acc += (pred == lbls).sum()
            predicted = out.argmax(dim=1)
            acc += torch.eq(predicted, lbls).sum().float().item()
    return loss.item() / count, acc / count


def validate_version_2(net, loader):

    loss_fn = torch.nn.NLLLoss()
    net.eval()
    corrected_value, loss = 0, 0
    total = len(loader.dataset)
    for feat, label in tqdm(loader, file=sys.stdout):

        out, lbls = hp.to_default_device(feat, label)
        with torch.no_grad():
            nn = net(out)
            predicted = nn.argmax(dim=1)
        corrected_value += torch.eq(predicted, lbls).sum().float().item()

    return corrected_value / total
