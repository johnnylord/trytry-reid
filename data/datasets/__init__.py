import sys

from .market1501 import *
from .dukemtmcreid import *
from .mnist import *

_dataset_classes = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'mnist': MNIST,
    }

def get_dataset_cls(name):
    return _dataset_classes[name]
