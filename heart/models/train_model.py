import sys

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


def train_epoch(model, train_loader, ep, lr=0.01, optim=None, loss_fn=None):
    bar = tqdm(train_loader, file=sys.stdout)
    acc, total_loss = 0, 0
    total = len(train_loader.dataset)
    for _, (feat, labels) in enumerate(bar):
        out, lbls = hp.to_default_device(feat, labels)
        model.train()
        output = model(out)
        loss = loss_fn(output, lbls)
        pred = output.argmax(dim=1)
        acc += torch.eq(pred, lbls).sum().float().item()
        optim.zero_grad()
        loss.backward()
        total_loss += loss.item()
        bar.desc = f"train epoch[{ep+1}/{ep}] loss:{loss:.3f}"
        optim.step()

    return total_loss / total, acc / total


def train():
    res = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    data_paths = fetch_data()
    dl_paths = data_loader(data_paths["train"], data_paths["test"], batch_size=1000)
    train_loader, val_loader, test_loader = dl_paths["train"], dl_paths["val"], dl_paths["test"]

    model = CNN(5).to(hc.DEFAULT_DEVICE)
    optim = hc.optim["Adam"](model.parameters(), lr=hc.lr)
    loss_fn = hc.loss["CEL"]()
    for ep in range(20):
        tl, ta = train_epoch(model, train_loader, ep, hc.lr, optim, loss_fn)
        vl, va = validate(model, val_loader, loss_fn=loss_fn)
        log.info(f"Epoch {ep:2}, Train acc={ta:.5f}, Val acc={va:.5f}, Train loss={tl:.9f}, Val loss={vl:.9f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)

    torch.save(model.state_dict(), f"{hc.DATASET_DIR}model.mdl")
    return (res, test_loader, model, loss_fn)
