import sys

import torch
from tqdm import tqdm

from heart.core import hp


def validate(model, loader, loss_fn):
    model.eval()
    correct, loss = 0, 0
    total = len(loader.dataset)
    val_bar = tqdm(loader, file=sys.stdout)
    for x, y in val_bar:
        x, y = hp.to_default_device(x, y)
        with torch.no_grad():
            logits = model(x)
            predicted = logits.argmax(dim=1)
            loss += loss_fn(logits, y)
            correct += torch.eq(predicted, y).sum().float().item()
    return loss.item() / total, correct / total
