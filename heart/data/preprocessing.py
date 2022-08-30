import os

from icecream import ic

from heart.constants import DATASET_DIR

datasets = frozenset(
    os.path.join(dirname, filename) for dirname, _, filenames in os.walk(DATASET_DIR) for filename in filenames)
