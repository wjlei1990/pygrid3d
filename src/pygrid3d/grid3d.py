from __future__ import print_function, division
import os
import numpy as np
from __init__ import logger
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from measurebase import MeasureBase


class Grid3d(MeasureBase):
    """
    Class that handle the grid search solver for origin time and moment scalar
    """
    def __init__(self, cmtsource, data_container, config):
        MeasureBase.__init__(cmtsource, data_container, config)

        self.new_cmtsource = None

        self.t00_best = None
        self.t00_misfit = None
        self.t00_array = None

        self.m00_best = None
        self.m00_misfit = None
        self.m00_array = None

    def search(self):

        self.setup_weight()

        if self.config.origin_time_inversion:
            self.grid_search_origin_time()

        if self.config.energy_inversion:
            self.grid_search_energy()

    def grid_search_origin_time(self):

        logger.info("Origin time grid search")
        tshift_dict = self.calculate_tshift()

        t00_s = self.config.t00_s
        t00_e = self.config.t00_e
        dt00 = self.config.dt00_over_dt * \
            self.window[0].datalist['obsd'].stats.delta
        logger.info("Grid search time start and end: [%8.3f, %8.3f]"
                    % (t00_s, t00_e))
        logger.info("Grid search time interval:%10.3f" % dt00)

        t00_array = np.arange(t00_s, t00_e+dt00, dt00)
        nt00 = t00_array.shape[0]

        final_misfit = {}
        for tag in self.weight_dict:
            final_misfit[tag] = {}
            final_misfit[tag]['raw'] = np.zeros(nt00)
            final_misfit[tag]['weight'] = np.zeros(nt00)

        for i in range(nt00):
            t00 = t00_array[i]
            new_td = {}
            for tag in tshift_dict:
                new_td[tag] = tshift_dict[tag] + t00
            _misfit = self.ensemble_result_for_tshift(new_td)
            for tag in _misfit:
                final_misfit[tag]['raw'][i] = _misfit[tag]['raw']
                final_misfit[tag]['weight'][i] = _misfit[tag]['weight']

        # find minimum
        if self.config.weight_data:
            weight_tag = "weight"
        else:
            weight_tag = "raw"

        min_idx = final_misfit['all'][weight_tag].argmin()
        t00_best = t00_array[min_idx]

        logger.info("Minimum t00(relative to cmt origin time): %6.3f"
                    % t00_best)

        self.t00_best = t00_best
        self.t00_array = t00_array
        self.t00_misfit = final_misfit

    def grid_search_energy(self):

        logger.info('Energy grid Search')

        m00_s = self.config.m00_s
        m00_e = self.config.m00_e
        dm00 = self.config.dm00
        logger.info("Grid search energy start and end: [%6.3f, %6.3f]"
                    % (m00_s, m00_e))
        logger.info("Grid search energy interval: %6.3f" % dm00)

        m00_array = np.arange(m00_s, m00_e+dm00, dm00)
        nm00 = m00_array.shape[0]

        final_misfit = {}
        for tag in self.weight_dict:
            final_misfit[tag] = {}
            final_misfit[tag]['weight'] = np.zeros(nm00, 4)
            final_misfit[tag]['raw'] = np.zeros(nm00, 4)

        for i in range(nm00):
            m00 = m00_array[i]
            dlnA_dict, cc_amp_dict, kai_dict = \
                self.calculate_misfit_for_m00(m00)
            _misfit_dict = self.ensemble_result_for_energy(
                dlnA_dict, cc_amp_dict, kai_dict)
            for tag in _misfit_dict:
                final_misfit[tag]['raw'][i][0] = \
                    _misfit_dict[tag]['dlnA']['raw']
                final_misfit[tag]['weight'][i][0] = \
                    _misfit_dict[tag]['dlnA']['weight']

                final_misfit[tag]['raw'][i][1] = \
                    _misfit_dict[tag]['cc_amp']['raw']
                final_misfit[tag]['weight'][i][1] = \
                    _misfit_dict[tag]['cc_amp']['weight']

                final_misfit[tag]['raw'][i][2] = \
                    _misfit_dict[tag]['dlnA']['raw']
                final_misfit[tag]['weight'][i][2] = \
                    _misfit_dict[tag]['dlnA']['weight']

        if self.config.weight_data:
            weight_tag = "weight"
        else:
            weight_tag = "raw"

        misfit_mode = self.config.energy_misfit_mode
        for tag in self.weight_dict:
            if misfit_mode == 'power_ratio':
                final_misfit[tag][weight_tag][:, 3] = \
                    final_misfit[tag][weight_tag][:, 0]
            elif misfit_mode == 'cc_amplitude':
                final_misfit[tag][weight_tag][:, 3] = \
                    final_misfit[tag][weight_tag][:, 1]
            elif misfit_mode == 'waveform':
                final_misfit[tag][weight_tag][:, 3] = \
                    final_misfit[tag][weight_tag][:, 2]
            elif misfit_mode == "power_and_cc":
                final_misfit[tag][weight_tag][:, 3] = \
                    0.25 * final_misfit[tag][weight_tag][:, 0] + \
                    0.75 * final_misfit[tag][weight_tag][:, 1]
            else:
                raise NotImplementedError("Not implemented")

        # find minimum
        min_idx = final_misfit['all'][weight_tag].argmin()
        m00_best = m00_array[min_idx]

        logger.info("best m00: %6.3f" % m00_best)
        self.m00_best = m00_best
        self.m00_array = m00_array
        self.m00_misfit = final_misfit

    def plot_summary(self, outputdir="."):

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if self.config.origin_time_inversion:
            self.plot_origin_time_summary(outputdir=outputdir)
        if self.config.energy_inversion:
            self.plot_energy_summary(outputdir=outputdir)

    def plot_origin_time_summary(self, outputdir="."):
        """
        Plot histogram and misfit curve of origin time result

        :param outputdir:
        :return:
        """
        # histogram
        nrows = len(self.bin_category.keys())
        ncols = 1
        figname = "%s.time_grid_search_histogram.png" \
            % self.cmtsource.eventname
        figname = os.path.join(outputdir, figname)

        # prepare the dict
        stats_before = {}
        stats_after = {}
        for window in self.window:
            tag = window.tag['obsd']
            if tag not in stats_before.keys():
                stats_before[tag] = []
            for _idx in range(window.num_wins):
                stats_before[tag].append(window.tshift[_idx])
        stats_after = stats_before.copy()
        for key in stats_before.keys():
            stats_before[key] = np.array(stats_before[key])
            stats_after[key] = np.array(stats_after[key])
            stats_after[key] = stats_after[key] - self.t00_best

        entry_array = ['tshift']
        nrows = len(self.bin_category.keys())
        ncols = len(entry_array)
        plt.figure(figsize=(5*ncols, 5*nrows))
        G = gridspec.GridSpec(nrows, ncols)
        tag_array = stats_before.keys()
        for tag_idx, tag in enumerate(tag_array):
            for entry_idx, entry in enumerate(entry_array):
                self.plot_histogram_one_entry(
                    G[tag_idx, entry_idx], tag, entry,
                    stats_before[tag], stats_after[tag])
        print("Grid search time shift histogram figure: %s" % figname)
        plt.savefig(figname)

        # plot misfit curve
        figname = "%s.time_grid_search_misfit.png" % self.cmtsource.eventname
        figname = os.path.join(outputdir, figname)
        plt.figure()
        plt.plot(self.t00_array, self.t00_misfit)
        plt.grid()
        print("Grid search time shift misfit figure: %s" % figname)
        plt.savefig(figname)

    def plot_energy_summary(self, outputdir="."):
        """
        Plot histogram of dlnA

        :param outputdir:
        :return:
        """
        figname = "%s.energy_grid_search_histogram.png" \
            % self.cmtsource.eventname
        figname = os.path.join(outputdir, figname)

        # prepare the dict
        stats_before = {}
        stats_after = {}
        for window in self.window:
            tag = window.tag['obsd']
            obsd = window.datalist['obsd']
            synt = window.datalist['synt']
            synt_new = synt.copy()
            synt_new.data = synt_new.data * self.m00_best
            if tag not in stats_before.keys():
                stats_before[tag] = []
                stats_after[tag] = []
            [v1, d1, nshift1, cc1, dlnA1, cc_amp_value1] = \
                self.calculate_var_one_trace(obsd, synt, window.win_time)
            [v2, d2, nshift2, cc2, dlnA2, cc_amp_value2] = \
                self.calculate_var_one_trace(obsd, synt_new, window.win_time)
            for _idx in range(window.num_wins):
                stats_before[tag].append([v1[_idx]/d1[_idx], cc1[_idx],
                                          dlnA1[_idx], cc_amp_value1[_idx]])
                stats_after[tag].append([v2[_idx]/d2[_idx], cc2[_idx],
                                         dlnA2[_idx], cc_amp_value2[_idx]])
        for key in stats_before.keys():
            stats_before[key] = np.array(stats_before[key])
            stats_after[key] = np.array(stats_after[key])

        entry_array = ['Kai', 'CC', 'Power Ratio', 'CC AMP']
        nrows = len(self.bin_category.keys())
        ncols = len(entry_array)
        plt.figure(figsize=(5*ncols, 5*nrows))
        G = gridspec.GridSpec(nrows, ncols)
        tag_array = stats_before.keys()
        for tag_idx, tag in enumerate(tag_array):
            for entry_idx, entry in enumerate(entry_array):
                self.plot_histogram_one_entry(
                    G[tag_idx, entry_idx], tag, entry,
                    stats_before[tag][:, entry_idx],
                    stats_after[tag][:, entry_idx])
        print("Grid search energy histogram figure: %s" % figname)
        plt.savefig(figname)

        # plot misfit curve
        plt.figure()
        figname = "%s.energy_grid_search_misfit.png" % self.cmtsource.eventname
        figname = os.path.join(outputdir, figname)
        plt.plot(self.m00_array, self.m00_misfit)
        plt.grid()
        print("Grid search energy misfit curve figure: %s" % figname)
        plt.savefig(figname)

    @staticmethod
    def plot_histogram_one_entry(pos, tag, entry, value_before, value_after):
        plt.subplot(pos)
        plt.xlabel(entry)
        plt.ylabel(tag)
        ax_min = min(min(value_after), min(value_before))
        ax_max = max(max(value_after), max(value_after))
        if entry in ['tshift', 'CC AMP', 'Power Ratio']:
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max
        binwidth = (ax_max - ax_min) / 15
        plt.hist(value_before,
                 bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
                 facecolor='blue', alpha=0.3)
        plt.hist(value_after,
                 bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
                 facecolor='green', alpha=0.5)
