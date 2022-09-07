import torch
from loguru import logger as log

from heart.core import hc
from heart.data.hb import HeartBeatModify
from heart.models.autoencoder import AutoEncoder
from heart.models.cnn import CNNConv1d
from heart.models.train_model import train
from heart.models.validation import validate
from heart.visualization.visualize import show_data


#  TODO: (vsedov) (18:03:55 - 07/09/22): Refactor this, idk, there has to be a
#  better way of dealing withthis
def conv_1d():
    model = CNNConv1d(1, 5).to(hc.DEFAULT_DEVICE)
    return {
        "model": model,
        "lr": 3e-4,
        "optim": hc.optim["Adam"](model.parameters(), lr=3e-4),
        "loss_fn": hc.loss["NLLL"](),
        "epoch": 50,
        "name": "model_ecg_heartbeat_categorization_1_cnn_conv1d",
        "type": "cnn"
    }


def auto_encoder():
    model = AutoEncoder().to(hc.DEFAULT_DEVICE)
    return {
        "model": model,
        "lr": 3e-4,
        "optim": hc.optim["Adam"](model.parameters(), lr=3e-4, weight_decay=0),
        "loss_fn": hc.loss["MSE"](),
        "epoch": 50,
        "name": "model_ech_heartbeat_categorization_2_auto_encoder",
        "type": "auto_encoder"
    }


def validate_data(test_loader_1, test_loader_2, network_data):

    model, loss_fn, name, epoch, lr = network_data
    model.load_state_dict(torch.load(f"{hc.DATASET_DIR}{name}.pl"))

    test_loss_1, test_acc_1 = validate(model, test_loader_1, loss_fn)
    test_loss_2, test_acc_2 = validate(model, test_loader_2, loss_fn)

    log.info(test_loss_1, test_acc_1)
    log.info(test_loss_2, test_acc_2)


def train_data(model, train_loader, val_loader, loss_fn, optim, name, type, epoch, lr):
    model_data = train(model, train_loader, val_loader, loss_fn, optim, f'{name}.pt', epoch, lr, type)
    show_data(name, type, model_data)
    # we use this data to validate everything and to seee if its good
    return (model, loss_fn, name, epoch, lr, model_data)


def setup():
    #  TODO: (vsedov) (17:57:58 - 07/09/22): Refactor this,
    #  I do not like how this is coded: maybe used namedtuple :think:
    data = HeartBeatModify().data_loader()
    level_1_data = data["level_1"]
    level_2_data = data["level_2"]

    # So this is from DataLoader it self, so i think that would have to get refactored
    loader_1 = {level_1_data[i][0]: level_1_data[i][1]
                for i in range(3)}
    loader_2 = {level_2_data[i][0]: level_1_data[i][1]
                for i in range(3)}

    conv1d = {
        n_type: train_data(train_loader=data["train"], val_loader=data["valid"], **conv_1d())
        for n_type, data in {
            "conv1_loader_1": loader_1,
            "conv1_loader_2": loader_2
        }.items()
    }
    #
    # #  BUG: (vsedov) (18:08:04 - 07/09/22): Dimension error with  auto encoders
    # # autoencoder = {
    # #     data_type: train_data(train_loader=loader_1[0], val_loader=loader_2[1], **auto_encoder())
    # #     for data_type in ["ac_loader_1", "ac_loader_2"]
    # # }
    #
    # [validate_data(loader_1[2], loader_2[2], nn_type) for nn_type in [conv1d]]
