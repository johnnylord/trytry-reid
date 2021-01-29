import sys

from .resnet import *

def get_model_cls(name):
    return getattr(sys.modules[__name__], name)
