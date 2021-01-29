import sys

from .softmax import *
from .triplet import *
from .softtrip import *

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
