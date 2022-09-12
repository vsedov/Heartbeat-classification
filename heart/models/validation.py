import sys

import torch
from tqdm import tqdm

from heart.core import hc, hp
from heart.log import get_logger

log = get_logger(__name__)


def validate(model, loader, loss_fn):
    model.eval()
    correct, loss = 0, 0
    total = len(loader.dataset)
    val_bar = tqdm(loader, file=sys.stdout)
    for feat, lbl in val_bar:
        x, y = hp.to_default_device(feat, lbl)
        with torch.no_grad():
            logits = model(x)
            current_loss = loss_fn(logits, y)
            predicted = hp.filtered_acc(logits, loss_fn)

            predictred = logits.argmax()
            loss += current_loss.item() * feat.size(0)
            correct += torch.eq(predicted, y).sum().float().item()

    return loss / total, correct / total


def validate_ae(model, loader, loss_fn):
    model.eval()
    loss = 0, 0
    val_bar = tqdm(loader, file=sys.stdout)
    loss = []
    for i, (feat, lbl) in enumerate(val_bar):
        x, y = hp.to_default_device(feat, lbl)
        with torch.no_grad():
            logits = model(x)
            current_loss = loss_fn(logits, y)
            predicted = hp.filtered_acc(logits, loss_fn)
            val_bar.desc = f"val loss:{current_loss:.3f}"

            if i % 10:
                loss.append(float(current_loss))
    return loss
