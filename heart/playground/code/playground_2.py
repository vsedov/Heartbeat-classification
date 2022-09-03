import torch

from heart.log import get_logger

log = get_logger(__name__)

ft = torch.Tensor([[10, 20, 30, 40, 50]])
log.info(ft.shape)

log.info(ft.unsqueeze(0))
print("\n")

log.info(ft.unsqueeze(1))
