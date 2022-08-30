import os
from typing import Optional

import torch

TRACE_LOGGERS: list[str] = [f"!{__name__}"]
DEFAULT_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR: Optional[str] = os.path.dirname(__file__)
DATASET_DIR = os.path.join(DIR, "data/mitbih_ds/")
