import builtins

from torchinfo import summary

from heart.core import hc, hc_register_const, hp_register
from heart.log import get_logger
from heart.utils.constants import constants, constants_extra

log = get_logger(__name__)


def setup_constants():
    for name, const in constants().items():
        hc_register_const(name, const)


def setup_extra_constants():

    builtins.hc = hc
    for name, const in constants_extra().items():
        hc_register_const(name, const)


def setup_globals():

    @hp_register
    def to_default_device(*pointer):
        """Quick way to force default device to cuda"""
        for output in pointer:
            yield output.to(hc.DEFAULT_DEVICE)

    @hp_register
    def sum(network):
        """Show summary of a network"""
        summary(network)

    @hp_register
    def unpack(iterable):
        for _ in iterable:
            ...

    @hp_register
    def save(figure, file_name, title):
        file_path = f"{hc.DIR}reports/data/raw/"
        filename = f"{file_path}{file_name}"
        figure.suptitle(title)
        figure.savefig(f"{filename}.png")

    @hp_register
    def filtered_acc(network, loss_fn):
        loss_function_name = loss_fn.__class__.__name__
        log.info(loss_function_name)
        if loss_function_name == "MSELoss":
            return network.argmax()
        return network.argmax(dim=1)
