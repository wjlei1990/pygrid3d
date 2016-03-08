from __future__ import (print_function, division, absolute_import)
import logging

__version__ = "0.1.0"

# setup the logger
logger = logging.getLogger("pygrid3d")
logger.setLevel(logging.INFO)
logger.propagate = 0

ch = logging.StreamHandler()
# Add formatter
FORMAT = "%(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

from .grid3d import Grid3d            # NOQA
from .statswindow import StatsWindow  # NOQA
