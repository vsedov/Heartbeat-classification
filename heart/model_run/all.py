from collections import ChainMap

from heart.data.hb import HeartBeatModify
from heart.model_run.config import auto_encoder, conv_1d, train_data, validate_data


def setup_all():
    hb = HeartBeatModify()
    auto_encoder_data = hb.auto_encoder_dataset()
    conv_data = hb.data_loader()

    conv_data = {level_type: ChainMap(*conv_data[level_type])
                 for level_type in ["level_1", "level_2"]}
    model_data = {
        f"{lt}_{network.name}":
        [pd := (train_data(lt, network, ld["train"], ld["valid"])),
         validate_data(ld["test"], pd)]
        for network in [conv_1d()] for lt, ld in conv_data.items()
    }
    ae_data_loader = {
        f"{lt}_{network.name}": [pd := (train_data(lt, network, ld, ld))]
        for lt, ld in auto_encoder_data.items() for network in [auto_encoder()]
    }

    return {
        **ae_data_loader,
        **model_data
    }
