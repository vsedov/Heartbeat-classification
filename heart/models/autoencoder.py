import torch.nn as nn

from heart.utils.defaults.model import DefaultModel


#  TODO: (vsedov) (19:55:29 - 03/09/22): Run mode.to(torch.double)
#  Though another option is to call self.double(), but im not sure if that
#  is valid enough though.
class AutoEncoder(DefaultModel):

    # Note : This will be a network , AutoEncoder ,
    # I will be testing  a new activation function known as SELU
    # Very interesting.
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(187, 100), nn.SELU(), nn.Linear(100, 40), nn.SELU(), nn.Linear(40, 20))

        self.decoder = nn.Sequential(
            nn.Linear(20, 40), nn.SELU(), nn.Linear(40, 100), nn.SELU(), nn.Linear(100, 187), nn.Sigmoid())
        # SELU Offers automatic normalisation, im curious to see how th eresults would compare with a normal relu
        # function

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
