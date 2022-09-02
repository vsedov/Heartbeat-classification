import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from heart.core import hc, hp
from heart.data.getdata import fetch_data
from heart.data.preprocessing import data_loader
from heart.log import get_logger
from heart.models.cnn import CNN
from heart.models.validation import validate

log = get_logger(__name__)
# Introduce a manual seed : one of the best seeds out there
# based on paper:
# https://arxiv.org/abs/2109.08203
torch.manual_seed(3407)

log.info(f"Current device being used is {hc.DEFAULT_DEVICE} on file\n{__file__}")

torch.manual_seed(3407)


def train_epoch(model, train_loader, lr=0.01, optim=None, loss_fn=None):

    total_loss, acc, count = 0, 0, 0
    for _, (feat, labels) in enumerate(tqdm(train_loader, file=sys.stdout)):
        optim.zero_grad()
        out, lbls = hp.to_default_device(feat, labels)
        model.train()
        out = model(out)
        loss = loss_fn(out, lbls)
        loss.backward()
        optim.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == lbls).sum()
        count += len(labels)
    return loss.item() / count, acc.item() / count


def train():
    res = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    global_batch_size = 1000
    data_paths = fetch_data()
    dl_paths = data_loader(data_paths["train"], data_paths["test"], batch_size=global_batch_size)
    train_loader, val_loader, test_loader = dl_paths["train"], dl_paths["val"], dl_paths["test"]

    model = CNN(5).to(hc.DEFAULT_DEVICE)
    optim = hc.optim["Adam"](model.parameters(), lr=hc.lr)
    loss_fn = hc.loss["CEL"]()

    for ep in range(hc.epochs):
        tl, ta = train_epoch(model, train_loader, optim=optim, lr=hc.lr, loss_fn=loss_fn)
        vl, va = validate(model, val_loader, loss_fn=loss_fn)
        log.info(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)

    #  TODO: (vsedov) (13:28:08 - 02/09/22): give path using hc
    torch.save(model.state_dict(), 'model.mdl')
    return (res, test_loader)
