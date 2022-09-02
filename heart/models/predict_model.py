import torch

from heart.core import hc
from heart.log import get_logger
from heart.models.train_model import train
from heart.models.validation import validate
from heart.visualization.visualize import show_data

log = get_logger(__name__)


def setup():

    loss_fn = hc.loss["CEL"]()
    hist, test_data, model = train()
    if hc.show_data:
        show_data(hist)

    model_path = f"{hc.DATASET_DIR}model.mdl"
    model.load_state_dict(torch.load(model_path))

    test_loss, test_acc = validate(model, test_data, loss_fn)
    log.info(test_acc)
    log.info(test_loss)
