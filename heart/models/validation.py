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
            loss += current_loss.item() * feat.size(0)
            if loss_fn.__class__.__name__ != "MSELoss":
                correct += torch.eq(predicted, y).sum().float().item()
            else:
                correct = (logits - y).abs().sum().item()

    return loss / total, correct / total
