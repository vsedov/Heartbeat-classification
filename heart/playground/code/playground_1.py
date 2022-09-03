import numpy as np
import torch

from heart.log import get_logger

log = get_logger(__name__)

leng = 5

x = np.array([[i + y for y in range(leng)] for i in range(leng)])
log.info(x)
log.info(torch.from_numpy(x[:, -1]).float())

test_string = "ting_test"
container = {"test", "train", "abnormal"}

test_string = test_string.split("_")[-1]
print(test_string)
