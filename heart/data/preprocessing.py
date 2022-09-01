import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from heart.core import hc
from heart.log import get_logger

log = get_logger(__name__)


def to_csv(*path):
    """
    to_csv convert path to cv
    *path
        Path to any valid csv to be yieled as
        torch readable data.

    Yields:
        Tuple[TensorDataset(), path.data.length]
    """
    for files in tqdm(path):
        data = pd.read_csv(files, header=None).to_numpy()
        yield (
            TensorDataset(
                torch.from_numpy(data[:, -1]).float(),
                torch.from_numpy(data[:, -1]).long(),
            ), data.shape[0])


def data_loader(train_path, test_path, batch_size=hc.BATCH_SIZE, validation_factor=0.01):
    """
    Dataloader : load data and returns Dataloader from the torch lib

    Parameters
    ----------
    train_path : Train Path for trainable csv data
        Trainable dataset : pandas used to convert data to numpy / torch
    test_path : Test Path for trainable trest csv data
        Test data: Same Principle as train_path : has to be a csv ds
    batch_size : Batch size
        Default batch size : 64 defined @constants.BATCH_SIZE
    validation_factor : Validation Factor
        Value of drop of with respect to the training length - allows us to define
        how much data would be required to be split and how much validation is required.

    Returns
    -------
    Dictionary of Dataloader object from torch
        {
        train: DataLoader(dataset),
        val: DataLoader(dataset)
        test: DataLoader(dataset)
        }
    """
    (train_data, train_len), (test_data, _) = to_csv(train_path, test_path)
    val_len = int(train_len * validation_factor)
    train_len -= val_len
    train_dataset, val_dataset = random_split(train_data, [train_len, val_len])

    # yapf: disable
    return {
        type_name: DataLoader(x, **{
            "batch_size": hc.BATCH_SIZE,
            "shuffle": True
        })
        for type_name, x in {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_data
        }.items()
    }
