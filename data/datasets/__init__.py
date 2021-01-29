import sys

from .market1501 import *
from .dukemtmcreid import *

_dataset_classes = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    }

def get_dataset_cls(name):
    return _dataset_classes[name]
