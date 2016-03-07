#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All the constants used in pycmt3d
"""

import numpy as np

# Mathmatical constants
PI = np.pi

# Number of regions for azimuthal weighting
NREGIONS = 10

# Earth's radius for depth scaling
R_EARTH = 6371  # km

# subset ratio of bootstrap
BOOTSTRAP_SUBSET_RATIO = 0.4