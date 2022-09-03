import torch

from heart.core import hc
from heart.log import get_logger
from heart.models.train_model import train
from heart.models.validation import validate
from heart.visualization.visualize import show_data, show_validation_loss

log = get_logger(__name__)


def setup_cnn():

    hist, test_data, model, loss_fn = train()
    if hc.show_data:
        show_data("cnn", hist)
    model.load_state_dict(torch.load(f"{hc.DATASET_DIR}model.mdl"))

    test_loss, test_acc = validate(model, test_data, loss_fn)
    show_validation_loss({
        "test_loss": test_loss,
        "test_acc": test_acc
    })
