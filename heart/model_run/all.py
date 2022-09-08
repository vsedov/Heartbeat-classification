from collections import ChainMap

import torch
from attr import define
from loguru import logger as log

from heart.core import hc
from heart.data.hb import HeartBeatModify
from heart.models.autoencoder import AutoEncoder
from heart.models.cnn import CNNConv1d
from heart.models.train_model import train
from heart.models.validation import validate
from heart.utils.defaults.model import DefaultModel
from heart.visualization.visualize import show_data


@define
class NetworkDefine:
    model: DefaultModel
    lr: int
    optim: torch
    loss_fn: torch
    epoch: int
    name: str
    network_type: str


def conv_1d():
    model = CNNConv1d(1, 5).to(hc.DEFAULT_DEVICE)

    return NetworkDefine(
        model, 3e-4, hc.optim["Adam"](model.parameters(), lr=3e-4), hc.loss["NLLL"](), 30,
        "model_ecg_heartbeat_categorization_1_cnn_conv1d", "cnn")


def auto_encoder():
    model = AutoEncoder().to(hc.DEFAULT_DEVICE)
    return NetworkDefine(
        model, 3e-4, hc.optim["Adam"](model.parameters(), lr=3e-4, weight_decay=0), hc.loss["MSE"](), 50,
        "model_ech_heartbeat_categorization_2_auto_encoder", "AE")


def validate_data(test_loader, network_data):
    model, loss_fn, model_name, model_type, _ = network_data
    model.load_state_dict(torch.load(f"{hc.DATASET_DIR}{model_type}/{model_name}.mdl"))
    test_loss_1, test_acc_1 = validate(model, test_loader, loss_fn)
    log.info(f"\nTest_loss: {test_loss_1}  Test_accuracy: {test_acc_1}")
    return test_acc_1, test_loss_1


def train_data(level_type, network_class, train_loader, valid_loader):
    model = network_class.model
    loss_fn = network_class.loss_fn
    optim = network_class.optim
    model_name = network_class.name + "_" + level_type
    epoch = network_class.epoch
    lr = network_class.lr
    model_type = network_class.network_type

    # Need to add more epochs for this for level_2 epochs this would be a really fat if statement
    # So i would have to use a dictionary
    model_data = train(model, train_loader, valid_loader, loss_fn, optim, model_name, epoch, lr, model_type)
    show_data(model_name, model_type, model_data)
    return (model, loss_fn, model_name, model_type, model_data)


def setup():
    data = HeartBeatModify().data_loader()
    level_data = {level_type: ChainMap(*data[level_type])
                  for level_type in ["level_1", "level_2"]}

    model_data = {
        f"{lt}_{network.name}":
        [pd := (train_data(lt, network, ld["train"], ld["valid"])),
         validate_data(ld["test"], pd)]
        for lt, ld in level_data.items() for network in [conv_1d()]
    }
    return model_data
