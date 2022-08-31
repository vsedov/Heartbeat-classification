import os
from typing import Optional

import torch
from dotenv import load_dotenv

load_dotenv()
#  ╭────────────────────────────────────────────────────────────────────╮
#  │                                                                    │
#  │ log                                                                │
#  │                                                                    │
#  ╰────────────────────────────────────────────────────────────────────╯
TRACE_LOGGERS: list[str] = [f"!{__name__}"]

#  ╭────────────────────────────────────────────────────────────────────╮
#  │                                                                    │
#  │ torch                                                              │
#  │                                                                    │
#  ╰────────────────────────────────────────────────────────────────────╯
DEFAULT_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
DEFAULT_LOSS = torch.nn.LogSoftmax
DEFAULT_OPTIM = torch.optim.Adam
#  ╭────────────────────────────────────────────────────────────────────╮
#  │                                                                    │
#  │ Directory information                                              │
#  │                                                                    │
#  ╰────────────────────────────────────────────────────────────────────╯
DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DIR, "data/heartbeat/")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
