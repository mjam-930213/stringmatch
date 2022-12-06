import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from stringmatch import _version
__version__ = _version.__version__

from stringmatch.config.logging import configure_logger
configure_logger()