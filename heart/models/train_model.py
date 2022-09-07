import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from heart.core import hc, hp
from heart.log import get_logger
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
    model.train()
    for _, (feat, labels) in enumerate(bar):
        out, lbls = hp.to_default_device(feat, labels)
        # Error Model is out of shape , or something is wrong here.

        output = model(out.double())
        loss = loss_fn(output, lbls)
        pred = output.argmax(dim=1)
        acc += torch.eq(pred, lbls).sum().float().item()
        optim.zero_grad()
        loss.backward()
        total_loss += loss.item() * feat.size(0)
        bar.desc = f"train epoch[{ep+1}/{ep}] loss:{loss:.3f}"
        optim.step()

    return total_loss / total, acc / total


def train(model, train_loader, valid_loader, loss_fn, optim, model_name, epoch, lr, model_type):
    model = model.double()
    model.cuda()
    valid_loss_min = np.Inf  # track change in validation loss
    res = defaultdict(list)
    res["train_loss"]
    res["train_acc"]
    res["val_acc"]
    res["val_loss"]
    # [09/07/22 18:19:16] INFO     Epoch  3 / 51, Train acc=0.84425, Val acc=0.24940, Train loss=0.446057108, Val loss=224.813652271                                                 train_model.py:58
    for ep in range(1, epoch + 1):
        tl, ta = train_epoch(model, train_loader, ep, lr, optim, loss_fn)
        vl, va = validate(model, valid_loader, loss_fn=loss_fn)
        log.info(
            f"Epoch {ep:2} / {epoch+1}, Train acc={ta:.5f}, Val acc={va:.5f}, Train loss={tl:.9f}, Val loss={vl:.9f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
        if vl <= valid_loss_min:
            log.info(f'Validation loss decreased ({valid_loss_min:.6f} --> {vl:.6f}).  Saving model ...')
            torch.save(model.state_dict(), f"{hc.DATASET_DIR}{model_type}/{model_name}")
            valid_loss_min = vl
    return res
