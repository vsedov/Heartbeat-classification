import torch

from heart.core import hc
from heart.log import get_logger
from heart.models.cnn import CNNConv1d
from heart.models.train_model import train
from heart.models.validation import validate
from heart.visualization.visualize import show_data, show_validation_loss

log = get_logger(__name__)


def setup_cnn():

    model = CNNConv1d(5).to(hc.DEFAULT_DEVICE)
    lr = 3e-4
    optim = hc.optim["Adam"](model.parameters(), lr=lr)
    loss_fn = hc.loss["CEL"]()
    
    epoch = 50
    bd = 500
    hist, test_data, model, loss_fn = train(model, optim, loss_fn, epoch, lr, bd)
    if hc.show_data:
        show_data("cnn", hist)
        

    show_validation_loss({
        "test_loss": test_loss,
        "test_acc": test_acc
    })


setup_cnn()
