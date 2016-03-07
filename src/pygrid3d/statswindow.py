import os
import glob
import numpy as np
from __init__ import logger, logfilename
import math
import const
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from util import construct_taper
from user_defined_weighting_function import default_weight_function 


class MiniConfig(object):

    def __init__(self, normalize_category=True, weight_function=None):

        self.normalize_category = normalize_category
        if weight_function is None:
            self.weight_function = default_weight_function
        else:
            self.weight_function = weight_function


class StatsWindow(object):
    """
    Class that stats window
    """

    def __init__(self, cmtsource, data_container, config):
        self.cmtsource = cmtsource
        self.data_container = data_container
        self.config = config
        self.window = self.data_container.window
        self.nwins = self.data_container.nwins

        self.weight_array = np.zeros(self.nwins)

        # azimuth information
        self.naz_files = None
        self.naz_files_all = None
        self.naz_wins = None
        self.naz_wins_all = None

        # category bin
        self.bin_category = None

        self.misfit = dict()
        self.weight_dict = dict()
        self.tshift_dict = dict()
        self.dlnA_dict = dict()
        self.cc_amp_dict = dict()
        self.kai_dict = dict()

    def setup_weight(self, weight_mode="num_wins"):
        """
        Use Window information to setup weight.
        :returns:
        """
        logger.info("*" * 15)
        logger.info("Start weighting...")

        # first calculate azimuth and distance for each data pair
        self.prepare_for_weighting()
        # then calculate azimuth weighting
        for idx, window in enumerate(self.window):
            if weight_mode.lower() == "num_files":
                # weighted by the number of files in each azimuth bin
                self.setup_weight_for_location(window, self.naz_files,
                                               self.naz_files_all)
            else:
                # weighted by the number of windows in each azimuth bin
                self.setup_weight_for_location(window, self.naz_wins,
                                               self.naz_wins_all)

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
        setup weight from location information, including distance, 
        component and azimuth

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
        logger.debug("%s.%s.%s, num_win, dist, naz: %d, %.2f, %d", 
                     window.station, window.network,
                     window.component,
                     window.num_wins, window.dist_in_km, naz)

        mode = "uniform"
        #mode = "exponential"
        window.weight = window.weight * \
            self.config.weight_function(window.component, window.dist_in_km,
                                        naz, window.num_wins,
                                        dist_weight_mode=mode)

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
            #window.weight = window.weight/math.sqrt(num_cat)
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
            logger.debug("%s.%s.%s, weight: [%s]" 
                         % (window.network, window.station, window.component,
                            ', '.join(map(self._float_to_str, window.weight))))
            window.weight /= max_weight
            logger.debug("Updated, weight: [%s]" 
                         % (', '.join(map(self._float_to_str, window.weight))))

    def prepare_for_weighting(self):
        """
        Prepare necessary information for weighting, e.x., 
        calculating azimuth, distance and energty of a window.
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
            logger.info("Azimuth file bin: [%s]" 
                        % (', '.join(map(str, self.naz_files[key]))))
            logger.info("Azimuth win bin: [%s]" 
                        % (', '.join(map(str, self.naz_wins[key]))))

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
            tag = window.tag
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
                                    (window.station, window.network,            
                                     window.location, window.component))        
                obsd_trace = obsd.data[istart:iend]                             
                synt_trace = synt.data[istart:iend]                             
                max_cc, nshift = self._xcorr_win_(obsd_trace, synt_trace)       
                tshift = nshift * obsd.stats.delta                              
                window.tshift[win_idx] = tshift                                 
                array_idx += 1                

        tshift_dict = {'all':[]}
        for window in self.window:                                              
            tag = window.tag['synt']
            if tag not in tshift_dict.keys():
                tshift_dict[tag] = []
            for win_idx in range(window.num_wins):
                tshift_dict[tag].append(window.tshift[win_idx])
                tshift_dict['all'].append(window.tshift[win_idx])
        for tag, value in tshift_dict.iteritems():
            self.tshift_dict[tag] = np.array(tshift_dict[tag])

    def stats_event(self):

        print "****************"
        print "See detailed result in: %s\n" %logfilename

        self.setup_weight()

        weight_dict = {'all':[],}
        for window in self.window:
            tag = window.tag['synt']
            if tag not in weight_dict.keys():
                weight_dict[tag] = []
            for win_idx in range(window.num_wins):
                weight_dict[tag].append(window.weight[win_idx])
                weight_dict['all'].append(window.weight[win_idx])
        for tag, value in weight_dict.iteritems():
            self.weight_dict[tag] = np.array(weight_dict[tag])

        self.stats_tshift()

        self.stats_energy()

        self.ensemble_result()

    def ensemble_result(self):
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
                                                                                
        logger.info("Origin time search...")                                    
                                                                                
        self.calculate_tshift()                                                 

    def stats_energy(self):

        logger.info('Energy Search...')

        self.calculate_misfit_for_m00(1.00)

    def write_output_log(self, outputdir="."):
        eventname = self.cmtsource.eventname
        for tag in self.tshift_dict.keys():
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

    def calculate_misfit_for_m00(self, m00):

        for window in self.window:
            obsd = window.datalist['obsd']
            synt = window.datalist['synt']
            synt_new = synt.copy()
            synt_new.data = synt_new.data * m00
            [v, d, nshift, cc, dlnA, cc_amp_value] = \
                self.calculate_var_one_trace(obsd, synt_new, window.win_time)

            window.dlnA = dlnA
            window.cc_amp = cc_amp_value
            window.kai_array = v/d

        dlnA_dict = {'all':[]}
        cc_amp_dict = {'all':[]}
        kai_dict = {'all':[]}

        for window in self.window:
            tag = window.tag['synt']
            if tag not in dlnA_dict.keys():
                dlnA_dict[tag] = []
                cc_amp_dict[tag] = []
                kai_dict[tag] = []
            dlnA_dict[tag].extend(window.dlnA)
            dlnA_dict['all'].extend(window.dlnA)
            cc_amp_dict[tag].extend(window.cc_amp)
            cc_amp_dict['all'].extend(window.cc_amp)
            kai_dict[tag].extend(window.kai_array)
            kai_dict['all'].extend(window.kai_array)

        for tag in dlnA_dict.keys():
            self.dlnA_dict[tag] = np.array(dlnA_dict[tag])
            self.cc_amp_dict[tag] = np.array(cc_amp_dict[tag])
            self.kai_dict[tag] = np.array(kai_dict[tag])

    def calculate_var_one_trace(self, obsd, synt, win_time):
        """
        Calculate the variance reduction on a pair of obsd and 
        synt and windows

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
            idx_end = int(min(math.ceil(tend / obsd.stats.delta), 
                              obsd.stats.npts))

            istart_d, iend_d, istart, iend, nshift, cc, dlnA, cc_amp_value = \
                self.apply_station_correction(obsd, synt, idx_start, idx_end)

            taper = construct_taper(iend - istart)
            v1[_win_idx] = \
                np.sum(taper * (synt.data[istart:iend] - 
                                obsd.data[istart_d:iend_d]) ** 2)
            d1[_win_idx] = np.sum(taper * obsd.data[istart_d:iend_d] ** 2)
            nshift_array[_win_idx] = nshift
            cc_array[_win_idx] = cc
            dlnA_array[_win_idx] = dlnA
            cc_amp_value_array[_win_idx] = cc_amp_value
            #print "v1, idx:", v1[_win_idx], istart, iend, istart_d, \
            #iend_d, _win_idx, nshift
        return [v1, d1, nshift_array, cc_array, dlnA_array, 
                cc_amp_value_array]

    def apply_station_correction(self, obsd, synt, istart, iend):
        """
        Apply station correction on windows based one cross-correlation 
        time shift if config.station_correction

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
        cc_amp_value = \
            10*np.log10(np.sum(obsd[istart_d:iend_d] * synt[istart_s:iend_s])
                        / (synt[istart_s:iend_s] ** 2).sum())
        return istart_d, iend_d, istart_s, iend_s, nshift, cc, \
            dlnA, cc_amp_value

    def calculate_criteria(self, obsd, synt, istart, iend):
        """
        Calculate the time shift, max cross-correlation value and 
        energy differnce

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
