from __future__ import (print_function, division)
import os
import json
import numpy as np
from __init__ import logger
from user_defined_weighting_function import default_weight_function
from .measurebase import MeasureBase


class StatsConfig(object):

    def __init__(self, normalize_category=True, weight_function=None):

        self.normalize_category = normalize_category
        if weight_function is None:
            self.weight_function = default_weight_function
        else:
            self.weight_function = weight_function


class StatsWindow(MeasureBase):
    """
    Class that stats window
    """

    def __init__(self, cmtsource, data_container, config):
        if isinstance(config, StatsConfig):
            raise TypeError("Input config must be type of MiniConfig")

        MeasureBase.__init__(self, cmtsource, data_container, config)

        self.misfit = dict()
        self.weight_dict = dict()
        self.tshift_dict = dict()
        self.dlnA_dict = dict()
        self.cc_amp_dict = dict()
        self.kai_dict = dict()

    def stats_event(self):

        self.setup_weight()
        self.stats_tshift()
        self.stats_energy()

    def ensemble_result_for_categories(self):
        for tag in self.tshift_dict.keys():
            self.misfit[tag] = {}
            self.misfit[tag]['tshift'] = \
                {"raw": np.sum(self.tshift_dict[tag]),
                 "weight": np.sum(self.tshift_dict[tag] *
                                  self.weight_dict[tag])}
            self.misfit[tag]['dlnA'] = \
                {"raw": np.sum(self.dlnA_dict[tag]),
                 "weight": np.sum(self.dlnA_dict[tag] *
                                  self.weight_dict[tag])}
            self.misfit[tag]['cc_amp'] = \
                {"raw": np.sum(self.cc_amp_dict[tag]),
                 "weight": np.sum(self.cc_amp_dict[tag] *
                                  self.weight_dict[tag])}
            self.misfit[tag]['kai'] = \
                {"raw": np.sum(self.kai_dict[tag]),
                 "weight": np.sum(self.kai_dict[tag] *
                                  self.weight_dict[tag])}
            self.misfit[tag]['nwins'] = len(self.weight_dict[tag])

    def stats_tshift(self):
        logger.info('Stats on the time shift in windows')
        tshift_dict = self.calculate_tshift()
        self.tshift_summary = self.ensemble_result_for_tshift(tshift_dict)

        self.tshift_dict = tshift_dict

    def stats_energy(self):
        logger.info('Energy Search...')
        dlnA_dict, cc_amp_dict, kai_dict = \
            self.calculate_misfit_for_m00(1.00)
        self.energy_summary = self.ensemble_result_for_energy(
            dlnA_dict, cc_amp_dict, kai_dict)

        self.dlnA_dict = dlnA_dict
        self.cc_amp_dict = cc_amp_dict
        self.kai_dict = kai_dict

    def write_output_log(self, outputdir="."):
        eventname = self.cmtsource.eventname
        for tag in self.weight_dict:
            tshift_array = self.tshift_dict[tag]
            dlnA_array = self.dlnA_dict[tag]
            cc_amp_array = self.cc_amp_dict[tag]
            kai_array = self.kai_dict[tag]
            weight_array = self.weight_dict[tag]

            filename = os.path.join(outputdir, "%s.%s.data.log" %
                                    (eventname, tag))
            with open(filename, 'w') as fh:
                for idx in range(len(tshift_array)):
                    fh.write("%12.4f %12.5f %12.5f %10.5f %12.5f\n" %
                             (tshift_array[idx], dlnA_array[idx],
                              cc_amp_array[idx], kai_array[idx],
                              weight_array[idx]))

        filename = os.path.join(outputdir, "%s.tshift.summary.json"
                                % eventname)
        with open(filename, "w") as fh:
            json.dump(self.tshift_summary, fh, indent=2)

        filename = os.path.join(outputdir, "%s.energy.summary.json"
                                % eventname)
        with open(filename, "w") as fh:
            json.dump(self.energy_summary, fh, indent=2)
