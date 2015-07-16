#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration object for pycmt3d
"""

try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)

import const
from __init__ import logger
from user_defined_weighting_function import *

class Config(object):
    """
    Configuration for source inversion

    :param npar: number of parameters to be inverted
    :param dlocation: location perturbation when calculated perturbed synthetic data
    :param ddepth: depth perturbation
    :param dmoment: moment perturbation
    :param weight_data: bool value of weighting data
    :param weight_function: weighting function
    :param normalize_window: add window energy into the weighting term
    :param norm_mode: two modes: 1) "data_and_synt" 2) "data_only"
    :param station_correction: bool value of whether applies station correction
    :param zero_trace: bool value of whether applies zero-trace constraint
    :param double_couple: bool value of whether applied double-couple constraint
    :param lamda_damping: damping coefficient
    :param bootstrap: bool value of whether applied bootstrap method
    :param bootstrap_repeat: bootstrap iterations
    """

    def __init__(self, origin_time_inversion=True, t00_s=-5.0, t00_e=-5.0, dt00=1.00,
                 energy_inversion=True, m00_s=-0.05, m00_e=0.05, dm00=0.01,
                 weight_data=True, weight_function=None,
                 normalize_category=True,
                 bootstrap=True, bootstrap_repeat=300):

        self.origin_time_inversion = origin_time_inversion
        self.t00_s = t00_s
        self.t00_e = t00_e
        self.dt00 = dt00
        self.energy_inversion = energy_inversion
        self.e00_s = m00_s
        self.e00_e = m00_e
        self.de00 = dm00
        self.weight_data = weight_data
        if weight_function is not None:
            self.weight_function = weight_function
        else:
            self.weight_function = default_weight_function
        self.normalize_category = normalize_category

        self.par_name = const.PAR_LIST

        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat

        self.print_summary()

    def print_summary(self):
        """
        Print function of configuration

        :return:
        """
        logger.info("="*10 + "  Config Summary  " + "="*10)
        logger.info("Origin time inversion: %s" % self.origin_time_inversion)
        logger.info("Energy inversion: %s" % self.energy_inversion)
        logger.info("Weighting scheme")
        if self.weight_data:
            if self.weight_function == default_weight_function:
                logger.info("   Weighting data ===> Using Default weighting function")
            else:
                logger.info("   Weighting data ===> Using user-defined weighting function")
        else:
            logger.info("   No weighting applied")
