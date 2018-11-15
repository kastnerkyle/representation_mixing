floatX = "float32"
intX = "int32"
import os

# fix logging during travis testing
if os.environ.get('TRAVIS') != "true":
    import logging
    logging.getLogger('tensorflow').disabled = True

from .core import get_logger
from .core import scan
from .core import dot
from .core import get_params_dict
from .core import run_loop
from .nodes import make_numpy_weights
from .nodes import make_numpy_biases

