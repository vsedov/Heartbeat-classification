import os
from functools import lru_cache
from subprocess import CalledProcessError, check_output

import torch
from dotenv import load_dotenv

from heart.log import get_logger

log = get_logger(__name__)

load_dotenv()


@lru_cache(maxsize=1)
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
        "BATCH_SIZE": 64,
        "DEFAULT_LOSS": torch.nn.LogSoftmax,
        "DEFAULT_OPTIM": torch.optim.Adam,
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME"),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY"),
        #  REVISIT: (vsedov) (12:57:15 - 01/09/22): I am not a fan of hardcoding this
        "DIR": f"{root()}/heart/"
    }


def constants_extra():

    return {
        "DATASET_DIR": os.path.join(hc.DIR, "data/heartbeat/"),
    }
