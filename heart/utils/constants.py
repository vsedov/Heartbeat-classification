import os
from functools import lru_cache
from subprocess import CalledProcessError, check_output

import torch
from dotenv import load_dotenv

from heart.log import get_logger

log = get_logger(__name__)

load_dotenv()


def root():
    ''' returns the absolute path of the repository root '''
    try:
        base = check_output('git rev-parse --show-toplevel', shell=True)
    except CalledProcessError:
        raise IOError('Current working directory is not a git repository')
    return base.decode('utf-8').strip()


def constants():
    return {
        "DEFAULT_DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
        "BATCH_SIZE": 1000,
        "loss": {
            # https://neptune.ai/blog/pytorch-loss-functions
            # Mean Absolute Error loss(x,y) = |x-y|
            "MAE": torch.nn.L1Loss,
            # Mean Square Error loss(x,y) =pow(x-y, 2)
            "MSE": torch.nn.MSELoss,
            # Negative Log Likelihood - google it ...
            "NLLL": torch.nn.NLLLoss,
            # Cross Entropy Loss Function
            "CEL": torch.nn.CrossEntropyLoss,
        },
        "optim": {
            # https://neptune.ai/blog/pytorch-loss-functions
            # Mean Absolute Error loss(x,y) = |x-y|
            "Adam": torch.optim.Adam,
            "Sgd": torch.optim.SGD,
            "Adagrad": torch.optim.Adagrad,
        },
        "lr": 2e-3,
        "EPOCH": 100,
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME"),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY"),
        "DIR": f"{root()}/heart/",
        "show_data": True,
    }


def constants_extra():
    return {
        "DATASET_DIR": os.path.join(hc.DIR, "data/heartbeat/"),
    }
