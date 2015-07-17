import os
import glob
import numpy as np
from __init__ import logger, logfilename
import math
import const
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

class Grid3d(object):
    """
    Class that handle the grid search solver for origin time and moment scalar
    """

    def __init__(self, cmtsource, data_container, config):
        self.config = config
        self.cmtsource = cmtsource
        self.data_container = data_container
        self.window = self.data_container.window
        self.nwins = self.data_container.nwins

        self.weight_array = np.zeros(self.nwins)

        self.new_cmt_par = None
        self.new_cmtsource = None

        # azimuth information
        self.naz_files = None
        self.naz_files_all = None
        self.naz_wins = None
        self.naz_wins_all = None

        # category bin
        self.bin_category = None

        # nshift array
        self.tshift_array = np.zeros(self.nwins)
        self.dlnA_array = np.zeros(self.nwins)

        self.t00_best = None
        self.t00_misfit = None
        self.t00_array = None

        self.m00_best = None
        self.m00_misfit = None
        self.m00_array = None

    def setup_weight(self, weight_mode="num_wins"):
        """
        Use Window information to setup weight.
        :returns:
        """
        logger.info("*" * 15)
        logger.info("Start weighting...")
        if self.config.weight_data:
            # first calculate azimuth and distance for each data pair
            self.prepare_for_weighting()
            # then calculate azimuth weighting
            for idx, window in enumerate(self.window):
                if weight_mode.lower() == "num_files":
                    # weighted by the number of files in each azimuth bin
                    self.setup_weight_for_location(window, self.naz_files, self.naz_files_all)
                else:
                    # weighted by the number of windows in each azimuth bin
                    self.setup_weight_for_location(window, self.naz_wins, self.naz_wins_all)

                if self.config.normalize_category:
                    self.setup_weight_for_category(window)

            # normalization of data weights
            self.normalize_weight()

        # prepare the weight array
        self.weight_array = np.zeros([self.data_container.nwins])
        _idx = 0
        for window in self.window:
            for win_idx in range(window.num_wins):
                self.weight_array[_idx] = window.weight[win_idx]
                _idx += 1

    def setup_weight_for_location(self, window, naz_bin, naz_bin_all):
        """
        setup weight from location information, including distance, component and azimuth
        :param window:
        :param naz_bin:
        :param naz_bin_all:
        :return:
        """
        idx_naz = self.get_azimuth_bin_number(window.azimuth)
        if self.config.normalize_category:
            tag = window.tag['obsd']
            naz = naz_bin[tag][idx_naz]
        else:
            naz = naz_bin_all[idx_naz]
        logger.debug("%s.%s.%s, num_win, dist, naz: %d, %.2f, %d", window.station, window.network,
                    window.component,
                    window.num_wins, window.dist_in_km, naz)

        mode = "uniform"
        #mode = "exponential"
        window.weight = window.weight * self.config.weight_function(window.component, window.dist_in_km,
                                                                    naz, window.num_wins, dist_weight_mode=mode)

    def setup_weight_for_category(self, window):
        """
        Setup weight for each category if config.normalize_category
        window_weight = window_weight / N_windows_in_category
        :param window:
        :return:
        """
        if self.config.normalize_category:
            tag = window.tag['obsd']
            num_cat = self.bin_category[tag]
            window.weight = window.weight/num_cat

    def normalize_weight(self):
        """
        Normalize the weighting and make the maximum to 1
        :return:
        """
        max_weight = 0.0
        for window in self.window:
            max_temp = np.max(window.weight)
            if max_temp > max_weight:
                max_weight = max_temp

        logger.debug("Global Max Weight: %f" % max_weight)

        for window in self.window:
            logger.debug("%s.%s.%s, weight: [%s]" % (window.network, window.station, window.component,
                                                     ', '.join(map(self._float_to_str, window.weight))))
            window.weight /= max_weight
            logger.debug("Updated, weight: [%s]" % (', '.join(map(self._float_to_str, window.weight))))

    def prepare_for_weighting(self):
        """
        Prepare necessary information for weighting, e.x., calculating azimuth, distance and energty of a window.
        Also, based on the tags, sort window into different categories.
        :return:
        """
        for window in self.window:
            # calculate energy
            #window.win_energy(mode=self.config.norm_mode)
            # calculate location
            window.get_location_info(self.cmtsource)

        self.naz_files, self.naz_wins = self.calculate_azimuth_bin()
        # add all category together
        # if not weighted by category, then use total number
        self.naz_files_all = np.zeros(const.NREGIONS)
        self.naz_wins_all = np.zeros(const.NREGIONS)
        for key in self.naz_files.keys():
            self.naz_files_all += self.naz_files[key]
            self.naz_wins_all += self.naz_wins[key]
            logger.info("Category: %s" % key)
            logger.info("Azimuth file bin: [%s]" % (', '.join(map(str, self.naz_files[key]))))
            logger.info("Azimuth win bin: [%s]" % (', '.join(map(str, self.naz_wins[key]))))

        # stat different category
        bin_category = {}
        for window in self.window:
            tag = window.tag['obsd']
            if tag in bin_category.keys():
                bin_category[tag] += window.num_wins
            else:
                bin_category[tag] = window.num_wins
        self.bin_category = bin_category


    @staticmethod
    def get_azimuth_bin_number(azimuth):
        """
        Calculate the bin number of a given azimuth
        :param azimuth: test test test
        :return:
        """
        # the azimuth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / const.NREGIONS
        k = int(math.floor(azimuth / daz))
        if k < 0 or k > const.NREGIONS:
            if azimuth - 360.0 < 0.0001:
                k = const.NREGIONS - 1
            else:
                raise ValueError('Error bining azimuth')
        return k

    def calculate_azimuth_bin(self):
        """
        Calculate the azimuth and sort them into bins
        :return:
        """
        naz_files = {}
        naz_wins = {}
        for window in self.window:
            tag = window.tag['obsd']
            bin_idx = self.get_azimuth_bin_number(window.azimuth)
            if tag not in naz_files.keys():
                naz_files[tag] = np.zeros(const.NREGIONS)
                naz_wins[tag] = np.zeros(const.NREGIONS)
            naz_files[tag][bin_idx] += 1
            naz_wins[tag][bin_idx] += window.num_wins
        return naz_files, naz_wins

    def calculate_tshift(self):
        array_idx = 0
        for window in self.window:
            window.tshift = np.zeros(window.num_wins)
            for win_idx in range(window.num_wins):
                datalist = window.datalist
                obsd = datalist['obsd']
                synt = datalist['synt']
                npts = min(obsd.stats.npts, synt.stats.npts)
                win = [window.win_time[win_idx, 0], window.win_time[win_idx, 1]]

                istart = int(max(math.floor(win[0] / obsd.stats.delta), 1))
                iend = int(min(math.ceil(win[1] / obsd.stats.delta), npts))
                if istart > iend:
                    raise ValueError("Check window for %s.%s.%s.%s" %
                                    (window.station, window.network, window.location, window.component))
                obsd_trace = obsd.data[istart:iend]
                synt_trace = synt.data[istart:iend]
                max_cc, nshift = self._xcorr_win_(obsd_trace, synt_trace)
                tshift = nshift * obsd.stats.delta
                window.tshift[win_idx] = tshift
                self.tshift_array[array_idx] = tshift
                array_idx += 1

    def grid_search_source(self):

        print "****************"
        print "See detailed result in: %s\n" %logfilename

        self.setup_weight()

        if self.config.origin_time_inversion:
            self.grid_search_origin_time()

        if self.config.energy_inversion:
            self.grid_search_energy()

    def grid_search_origin_time(self):

        logger.info("Origin time search...")

        self.calculate_tshift()

        t00_s = self.config.t00_s
        t00_e = self.config.t00_e
        dt00 = self.config.dt00_over_dt * self.window[0].datalist['obsd'].stats.delta
        logger.info("Grid search dt00: %6.3f" % dt00)

        t00_array = np.arange(t00_s, t00_e+dt00, dt00)
        nt00 = t00_array.shape[0]
        misfit = np.zeros(nt00)

        for i in range(nt00):
            t00 = t00_array[i]
            misfit[i] = self.calculate_tshift_misfit(t00)

        # find minimum
        min_idx = misfit.argmin()
        t00_best = t00_array[min_idx]

        logger.info("minimum t00: %6.3f" % t00_best)
        self.t00_best = t00_best
        self.t00_misfit = misfit
        self.t00_array = t00_array

    def grid_search_energy(self):

        logger.info('Energy Search...')

        m00_s = self.config.m00_s
        m00_e = self.config.m00_e
        dm00 = self.config.dm00
        m00_array = np.arange(m00_s, m00_e+dm00, dm00)
        nm00 = m00_array.shape[0]
        misfit = np.zeros(nm00)

        for i in range(nm00):
            m00 = m00_array[i]
            dlnA_array, cc_amp_array, kai_array = self.calculate_misfit_for_m00(m00)
            if self.config.energy_misfit_function == 'energy':
                misfit[i] = np.sum((0.25 * dlnA_array ** 2 + 1.0 * cc_amp_array ** 2) * self.weight_array)
            else:
                misfit[i] = np.sum(kai_array * self.weight_array)

        # find minimum
        min_idx = misfit.argmin()
        m00_best = m00_array[min_idx]

        logger.info("best m00: %6.3f" % m00_best)
        self.m00_best = m00_best
        self.m00_misfit = misfit
        self.m00_array = m00_array

    def calculate_misfit_for_m00(self, m00):
        dlnA_array = []
        cc_amp_array = []
        kai_array = []
        total_idx = 0
        for window in self.window:
            obsd = window.datalist['obsd']
            synt = window.datalist['synt']
            synt_new = synt.copy()
            synt_new.data = synt_new.data * m00
            [v, d, nshift, cc, dlnA, cc_amp_value] = self.calculate_var_one_trace(obsd, synt_new, window.win_time)
            for win_idx in range(window.num_wins):
                dlnA_array.append(dlnA[win_idx])
                cc_amp_array.append(cc_amp_value[win_idx])
                kai_array.append(v[win_idx]/d[win_idx])
        return np.array(dlnA_array), np.array(cc_amp_array), np.array(kai_array)

    def calculate_var_one_trace(self, obsd, synt, win_time):
        """
        Calculate the variance reduction on a pair of obsd and synt and windows
        :param obsd: observed data trace
        :type obsd: :class:`obspy.core.trace.Trace`
        :param synt: synthetic data trace
        :type synt: :class:`obspy.core.trace.Trace`
        :param win_time: [win_start, win_end]
        :type win_time: :class:`list` or :class:`numpy.array`
        :return:  waveform misfit reduction and observed data energy [v1, d1]
        :rtype: [float, float]
        """
        num_wins = win_time.shape[0]
        v1 = np.zeros(num_wins)
        d1 = np.zeros(num_wins)
        nshift_array = np.zeros(num_wins)
        cc_array = np.zeros(num_wins)
        dlnA_array = np.zeros(num_wins)
        cc_amp_value_array = np.zeros(num_wins)
        npts = min(obsd.stats.npts, synt.stats.npts)
        for _win_idx in range(win_time.shape[0]):
            tstart = win_time[_win_idx, 0]
            tend = win_time[_win_idx, 1]
            idx_start = int(max(math.floor(tstart / obsd.stats.delta), 1))
            idx_end = int(min(math.ceil(tend / obsd.stats.delta), obsd.stats.npts))

            istart_d, iend_d, istart, iend, nshift, cc, dlnA, cc_amp_value = \
                self.apply_station_correction(obsd, synt, idx_start, idx_end)

            taper = self.construct_hanning_taper(iend - istart)
            v1[_win_idx] = np.sum(taper * (synt.data[istart:iend] - obsd.data[istart_d:iend_d]) ** 2)
            d1[_win_idx] = np.sum(taper * obsd.data[istart_d:iend_d] ** 2)
            nshift_array[_win_idx] = nshift
            cc_array[_win_idx] = cc
            dlnA_array[_win_idx] = dlnA
            cc_amp_value_array[_win_idx] = cc_amp_value
            # print "v1, idx:", v1[_win_idx], istart, iend, istart_d, iend_d, _win_idx, nshift
        return [v1, d1, nshift_array, cc_array, dlnA_array, cc_amp_value_array]

    def apply_station_correction(self, obsd, synt, istart, iend):
        """
        Apply station correction on windows based one cross-correlation time shift if config.station_correction
        :param obsd:
        :param synt:
        :param istart:
        :param iend:
        :return:
        """
        npts = min(obsd.stats.npts, synt.stats.npts)
        [nshift, cc, dlnA] = self.calculate_criteria(obsd, synt, istart, iend)
        istart_d = max(1, istart + nshift)
        iend_d = min(npts, iend + nshift)
        istart_s = istart_d - nshift
        iend_s = iend_d - nshift
        # recalculate the dlnA and cc_amp_value(considering the shift)
        dlnA = self._dlnA_win_(obsd[istart_d:iend_d], synt[istart_s:iend_s])
        cc_amp_value = 10*np.log10(np.sum(obsd[istart_d:iend_d] * synt[istart_s:iend_s]) / (synt[istart_s:iend_s] ** 2).sum())
        return istart_d, iend_d, istart_s, iend_s, nshift, cc, dlnA, cc_amp_value

    def calculate_criteria(self, obsd, synt, istart, iend):
        """
        Calculate the time shift, max cross-correlation value and energy differnce
        :param obsd: observed data trace
        :type obsd: :class:`obspy.core.trace.Trace`
        :param synt: synthetic data trace
        :type synt: :class:`obspy.core.trace.Trace`
        :param istart: start index of window
        :type istart: int
        :param iend: end index of window
        :param iend: int
        :return: [number of shift points, max cc value, dlnA]
        :rtype: [int, float, float]
        """
        obsd_trace = obsd.data[istart:iend]
        synt_trace = synt.data[istart:iend]
        max_cc, nshift = self._xcorr_win_(obsd_trace, synt_trace)
        dlnA = self._dlnA_win_(obsd_trace, synt_trace)

        return [nshift, max_cc, dlnA]

    def calculate_tshift_misfit(self, t00):
        misfit = np.sum(self.weight_array * (self.tshift_array - t00)**2)
        return misfit

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
        figname = "%s.time_grid_search_histogram.png" % self.cmtsource.eventname
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
                self.plot_histogram_one_entry(G[tag_idx, entry_idx], tag, entry, stats_before[tag], stats_after[tag])
        print "Grid search time shift histogram figure: %s" % figname
        plt.savefig(figname)

        # plot misfit curve
        figname = "%s.time_grid_search_misfit.png" % self.cmtsource.eventname
        figname = os.path.join(outputdir, figname)
        plt.figure()
        plt.plot(self.t00_array, self.t00_misfit)
        plt.grid()
        print "Grid search time shift misfit figure: %s" % figname
        plt.savefig(figname)

    def plot_energy_summary(self, outputdir="."):
        """
        Plot histogram of dlnA 

        :param outputdir:
        :return:
        """
        figname = "%s.energy_grid_search_histogram.png" % self.cmtsource.eventname
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
            [v1, d1, nshift1, cc1, dlnA1, cc_amp_value1] = self.calculate_var_one_trace(obsd, synt, window.win_time)
            [v2, d2, nshift2, cc2, dlnA2, cc_amp_value2] = self.calculate_var_one_trace(obsd, synt_new, window.win_time)
            for _idx in range(window.num_wins):
                stats_before[tag].append([v1[_idx]/d1[_idx], cc1[_idx], dlnA1[_idx], cc_amp_value1[_idx]])
                stats_after[tag].append([v2[_idx]/d2[_idx], cc2[_idx], dlnA2[_idx], cc_amp_value2[_idx]])
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
                self.plot_histogram_one_entry(G[tag_idx, entry_idx], tag, entry, stats_before[tag][:, entry_idx],
                                              stats_after[tag][:, entry_idx])
        print "Grid search energy histogram figure: %s" % figname
        plt.savefig(figname)

        # plot misfit curve
        plt.figure()
        figname = "%s.energy_grid_search_misfit.png" % self.cmtsource.eventname
        figname = os.path.join(outputdir, figname)
        plt.plot(self.m00_array, self.m00_misfit)
        plt.grid()
        print "Grid search energy misfit curve figure: %s" % figname
        plt.savefig(figname)

    @staticmethod
    def plot_histogram_one_entry(pos, tag, entry, value_before, value_after):
        ax = plt.subplot(pos)
        plt.xlabel(entry)
        plt.ylabel(tag)
        ax_min = min(min(value_after), min(value_before))
        ax_max = max(max(value_after), max(value_after))
        if entry in ['tshift', 'CC AMP', 'Power Ratio']:
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max
        binwidth = (ax_max - ax_min) / 15
        plt.hist(value_before, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth), facecolor='blue', alpha=0.3)
        plt.hist(value_after, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth), facecolor='green', alpha=0.5)

    @staticmethod
    def _xcorr_win_(obsd, synt):
        cc = np.correlate(obsd, synt, mode="full")
        nshift = cc.argmax() - len(obsd) + 1
        # Normalized cross correlation.
        max_cc_value = cc.max() / np.sqrt((synt ** 2).sum() * (obsd ** 2).sum())
        return max_cc_value, nshift

    @staticmethod
    def _dlnA_win_(obsd, synt):
        return 10 * np.log10(np.sum(obsd ** 2) / np.sum(synt ** 2))

    @staticmethod
    def construct_hanning_taper(npts):
        """
        Hanning taper construct
        :param npts: number of points
        :return:
        """
        taper = np.zeros(npts)
        #taper = np.ones(npts)
        #return taper
        for i in range(npts):
            taper[i] = 0.5 * (1 - math.cos(2 * np.pi * (float(i) / (npts - 1))))
        return taper

    @staticmethod
    def _float_to_str(value):
        """
        Convert float value to a specific precision string
        :param value:
        :return: string of the value
        """
        return "%.5f" % value

    @staticmethod
    def _float_array_to_str(array):
        """
        Convert float array to string
        :return:
        """
        string = "[  "
        for ele in array:
            string += "%10.3e  " % ele
        string += "]"
        return string
