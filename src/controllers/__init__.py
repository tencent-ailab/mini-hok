REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
