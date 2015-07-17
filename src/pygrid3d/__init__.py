from __future__ import (absolute_import)
import logging.config
import os

# setup the logger
logger = logging.getLogger("pycmt3d")
logger.setLevel(logging.INFO)
logger.propagate = 0

logfilename = "grid3d.log"
if os.path.exists(logfilename):
    os.remove(logfilename)

# ch = logging.StreamHandler()
ch = logging.FileHandler(logfilename, mode='w')
# Add formatter
FORMAT = "%(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)
