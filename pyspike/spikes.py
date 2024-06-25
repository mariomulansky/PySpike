# Module containing several function to load and transform spike trains
# Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

import numpy as np
import matplotlib.pyplot as plt
import pyspike as spk
from pyspike import SpikeTrain


############################################################
# spike_train_from_string
############################################################
def spike_train_from_string(s, edges, sep=' ', is_sorted=False):
    """ Converts a string of times into a  :class:`.SpikeTrain`.

    :param s: the string with (ordered) spike times.
    :param edges: interval defining the edges of the spike train.
                  Given as a pair of floats (T0, T1) or a single float T1,
                  where T0=0 is assumed.
    :param sep: The separator between the time numbers, default=' '.
    :param is_sorted: if True, the spike times are not sorted after loading,
                      if False, spike times are sorted with `np.sort`
    :returns: :class:`.SpikeTrain`
    """
    return SpikeTrain(np.fromstring(s, sep=sep), edges, is_sorted)


############################################################
# load_spike_trains_from_txt
############################################################
def load_spike_trains_from_txt(file_name, edges,
                               separator=' ', comment='#', is_sorted=False,
                               ignore_empty_lines=True):
    """ Loads a number of spike trains from a text file. Each line of the text
    file should contain one spike train as a sequence of spike times separated
    by `separator`. Empty lines as well as lines starting with `comment` are
    neglected. The `edges` represents the start and the end of the
    spike trains.

    :param file_name: The name of the text file.
    :param edges: A pair (T_start, T_end) of values representing the
                  start and end time of the spike train measurement
                  or a single value representing the end time, the
                  T_start is then assuemd as 0.
    :param separator: The character used to seprate the values in the text file
    :param comment: Lines starting with this character are ignored.
    :param sort: If true, the spike times are order via `np.sort`, default=True
    :returns: list of :class:`.SpikeTrain`
    """
    spike_trains = []
    with open(file_name, 'r') as spike_file:
        for line in spike_file:
            if not line.startswith(comment):  # ignore comments
                if len(line) > 1:
                    # ignore empty lines
                    spike_train = spike_train_from_string(line, edges,
                                                          separator, is_sorted)
                    spike_trains.append(spike_train)
                elif not(ignore_empty_lines):
                    # add empty spike train
                    spike_trains.append(SpikeTrain([], edges))
    return spike_trains


def import_spike_trains_from_time_series(file_name, start_time, time_bin,
                                         separator=None, comment='#'):
    """ Imports spike trains from time series consisting of 0 and 1 denoting
    the absence or presence of a spike. Each line in the data file represents
    one spike train.

    :param file_name: The name of the data file containing the time series.
    :param edges: A pair (T_start, T_end) of values representing the
                  start and end time of the spike train measurement
                  or a single value representing the end time, the
                  T_start is then assuemd as 0.
    :param separator: The character used to seprate the values in the text file
    :param comment: Lines starting with this character are ignored.

    """
    data = np.loadtxt(file_name, comments=comment, delimiter=separator)
    time_points = start_time + time_bin + np.arange(len(data[0, :]))*time_bin
    spike_trains = []
    for time_series in data:
        spike_trains.append(SpikeTrain(time_points[time_series > 0],
                                       edges=[start_time,
                                              time_points[-1]]))
    return spike_trains


############################################################
# save_spike_trains_to_txt
############################################################
def save_spike_trains_to_txt(spike_trains, file_name,
                             separator=' ', precision=8):
    """ Saves the given spike trains into a file with the given file name.
    Each spike train will be stored in one line in the text file with the times
    separated by `separator`.

    :param spike_trains: List of :class:`.SpikeTrain` objects
    :param file_name: The name of the text file.
    """
    # format string to print the spike times with given precision
    format_str = "{0:.%de}" % precision
    with open(file_name, 'w') as spike_file:
        for st in spike_trains:
            s = separator.join(map(format_str.format, st.spikes))
            spike_file.write(s+'\n')


############################################################
# merge_spike_trains
############################################################
def merge_spike_trains(spike_trains):
    """ Merges a number of spike trains into a single spike train.

    :param spike_trains: list of :class:`.SpikeTrain`
    :returns: spike train with the merged spike times
    """
    # concatenating and sorting with numpy is fast, it also means we can handle
    # empty spike trains
    merged_spikes = np.concatenate([st.spikes for st in spike_trains])
    merged_spikes.sort()
    return SpikeTrain(merged_spikes, [spike_trains[0].t_start,
                                      spike_trains[0].t_end])


############################################################
# generate_poisson_spikes
############################################################
def generate_poisson_spikes(rate, interval):
    """ Generates a Poisson spike train with the given rate in the given time
    interval

    :param rate: The rate of the spike trains
    :param interval: A pair (T_start, T_end) of values representing the
                     start and end time of the spike train measurement or
                     a single value representing the end time, the T_start
                     is then assuemd as 0. Auxiliary spikes will be added
                     to the spike train at the beginning and end of this
                     interval, if they are not yet present.
    :type interval: pair of doubles or double
    :returns: Poisson spike train as a :class:`.SpikeTrain`
    """
    try:
        T_start = interval[0]
        T_end = interval[1]
    except:
        T_start = 0
        T_end = interval
    # roughly how many spikes are required to fill the interval
    N = max(1, int(1.2 * rate * (T_end-T_start)))
    N_append = max(1, int(0.1 * rate * (T_end-T_start)))
    intervals = np.random.exponential(1.0/rate, N)
    # make sure we have enough spikes
    while T_start + sum(intervals) < T_end:
        # print T_start + sum(intervals)
        intervals = np.append(intervals,
                              np.random.exponential(1.0/rate, N_append))
    spikes = T_start + np.cumsum(intervals)
    spikes = spikes[spikes < T_end]
    return SpikeTrain(spikes, interval)

def reconcile_spike_trains(spike_trains):
    """ make sure that Spike trains meet PySpike rules
            In: spike_trains - a list of SpikeTrain objects
            Out: spike_trains - same list with some fixes:
              1) t_start and t_end are the same for every train
              2) The spike times are sorted
              3) No duplicate times in any train  
              4) spike times outside of t_start,t_end removed
    """
    ## handle sorting and uniqueness first (np.unique() does a sort)
    spike_trains = [SpikeTrain(np.unique(s.spikes), 
                               [s.t_start, s.t_end], 
                               is_sorted=True) for s in spike_trains]

    ## find global start and end times
    Starts = [s.t_start  for s in spike_trains]
    tStart = min(Starts)

    Ends   = [s.t_end for s in spike_trains]
    tEnd = max(Ends)

    ## remove spike times outside of the bounds
    Eps = 1e-6 #beware precision change
    for s in spike_trains:
        s.spikes = [t for t in s.spikes if t > tStart-Eps and t < tEnd+Eps]

    ## Apply start and end times to every train
    return [SpikeTrain(s.spikes, [tStart, tEnd], is_sorted=True) for s in spike_trains]

def reconcile_spike_trains_bi(spike_train1, spike_train2):
    """ fix up a pair of spike trains"""
    trains_in = [spike_train1, spike_train2]
    trains_out = reconcile_spike_trains(trains_in)
    return trains_out[0], trains_out[1]

############################################################
# create synfire
############################################################

def renorm(values, tmin, tmax, inside=0):
    if isinstance(values, list):
        all_values = np.concatenate(values)
        minas = np.min(all_values)
        maxas = np.max(all_values)
        num_trains = len(values)
        norm_values = []
        for trc in range(num_trains):
            if maxas > minas:
                if len(values[trc]) > 0:
                    if inside == 1:
                        if values[trc][0] == tmin:
                            norm_values.append(tmin + 1e-10)
                        elif values[trc][0] == tmax:
                            norm_values.append(tmax - 1e-10)
                        else:
                            norm_values.append((values[trc] - minas) / (maxas - minas) * (tmax - tmin - 2 * 1e-10) + tmin + 1e-10)
                    else:
                        norm_values.append((values[trc] - minas) / (maxas - minas) * (tmax - tmin) + tmin)
            else:
                if inside == 1:
                    if values[trc][0] == tmin:
                        norm_values.append(tmin + 1e-10)
                    elif values[trc][0] == tmax:
                        norm_values.append(tmax - 1e-10)
                    else:
                        norm_values.append(values[trc])
                else:
                    norm_values.append(values[trc])
        return norm_values
    else:
        minas = np.min(values)
        maxas = np.max(values)
        if maxas > minas:
            if inside == 1:
                if values[0] == tmin:
                    return tmin + 1e-10
                elif values[0] == tmax:
                    return tmax - 1e-10
                else:
                    return (values - minas) / (maxas - minas) * (tmax - tmin - 2 * 1e-10) + tmin + 1e-10
            else:
                return (values - minas) / (maxas - minas) * (tmax - tmin) + tmin
        else:
            if inside == 1:
                if values[0] == tmin:
                    return tmin + 1e-10
                elif values[0] == tmax:
                    return tmax - 1e-10
                else:
                    return values
            else:
                return values

def create_synfire(tmin, tmax, num_trains, num_synfire_events, num_inverse_events, overlap, shuffle, jitter, complete, background, order, plotting):
    refractoriness = 0.0001
    num_total_events = num_synfire_events + num_inverse_events
    num_total_spikes = num_trains * num_total_events

    if overlap > 0:
        distance_btw_events = (tmax - tmin) / overlap / ((num_total_events - 1) / overlap + 1)
    else:
        distance_btw_events = (tmax - tmin) / (num_total_events - 1)

    event_duration = distance_btw_events * overlap
    spike_time_diff = event_duration / (num_trains - 1)
    spikes = [[] for _ in range(num_trains)]

    for trc in range(num_trains):
        spikes[trc] = np.arange(tmin + trc * spike_time_diff, tmax + 1, distance_btw_events)
        
    original_shift = [x[0] for x in spikes]

    if shuffle > 0:
        num_shuffle_spikes = round(shuffle * num_trains)
        if num_shuffle_spikes > 1:
            for ec in range(num_total_events):
                dummy = np.random.permutation(num_shuffle_spikes)
                while any(dummy == np.arange(1, num_shuffle_spikes + 1)):
                    dummy = np.random.permutation(num_shuffle_spikes)
                rand_indy = np.random.permutation(num_trains)
                indy = rand_indy[:num_shuffle_spikes]
                event_spikes = [spikes[ind][ec] for ind in indy]
                for spc in range(num_shuffle_spikes):
                    spikes[indy[dummy[spc]]][ec] = event_spikes[spc]

    if num_inverse_events > 0:
        if num_inverse_events > num_synfire_events:
            for ec in range(num_inverse_events):
                dummy = [spikes[x][ec] for x in range(num_trains)]
                for trc in range(num_trains):
                    spikes[trc][ec] = dummy[num_trains - 1 - trc]
        else:
            for ec in range(num_synfire_events, num_total_events):
                dummy = [spikes[x][ec] for x in range(num_trains)]
                for trc in range(num_trains):
                    spikes[trc][ec] = dummy[num_trains - 1 - trc]

    if jitter > 0:
        mean_isi = np.mean(np.diff(spikes[0]))
        for trc in range(num_trains):
            spikes[trc] += np.random.randn(num_total_events) * mean_isi * jitter

    if complete < 1:
        num_total_spikes = num_trains * num_total_events
        num_events_spikes = round(num_total_spikes * complete)
        num_sel_spikes = np.ones(num_total_events, dtype=int) * (num_events_spikes // num_total_events)
        if num_events_spikes % num_total_events > 0:
            rp = np.random.permutation(num_total_events)
            rpi = rp[:num_events_spikes % num_total_events]
            num_sel_spikes[rpi] += 1
        indy = [[] for _ in range(num_total_events)]
        for ec in range(num_total_events):
            rand_indy = np.random.permutation(num_trains)
            indy[ec] = np.sort(rand_indy[:num_sel_spikes[ec]])
        for trc in range(num_trains):
            sel_events = [any(np.isin(x, trc)) for x in indy]
            spikes[trc] = [spikes[trc][i] for i, x in enumerate(sel_events) if x]

    if background > 0:
        num_exp_spikes = num_total_events * complete * (1 + background)
        if 1 / num_exp_spikes < 5 * refractoriness:
            refractoriness = 0.00001
            if 1 / num_exp_spikes < 5 * refractoriness:
                refractoriness = 0.000001
                if 1 / num_exp_spikes < 5 * refractoriness:
                    refractoriness = 0.0000001
                    if 1 / num_exp_spikes < 5 * refractoriness:
                        refractoriness = 0.00000001
        spikes6 = [[] for _ in range(num_trains)]
        num_background_spikes = round(background * num_total_spikes)
        num_sel_spikes = np.ones(num_trains, dtype=int) * (num_background_spikes // num_trains)
        if num_background_spikes % num_trains > 0:
            rp = np.random.permutation(num_trains)
            rpi = rp[:num_background_spikes % num_trains]
            num_sel_spikes[rpi] += 1
        for trc in range(num_trains):
            backy = np.random.rand(num_sel_spikes[trc]) * tmax
            spikes6[trc] = np.array([0, 0], dtype=float)
            wlc = 0
            while np.min(np.diff(spikes6[trc])) < refractoriness:
                if wlc == 0:
                    spikes6[trc] = np.sort(np.concatenate((spikes[trc], backy)))
                    wlc += 1
                else:
                    minp = np.argmin(np.diff(spikes6[trc]))
                    if spikes6[trc][minp] > refractoriness:
                        spikes6[trc][minp] -= refractoriness
                    else:
                        spikes6[trc][minp + 1] += refractoriness
        spikes = spikes6

    num_spikes = [len(x) for x in spikes]

    unnorm_spikes = spikes
    spikes = renorm(spikes, tmin, tmax, 0)
    indy = np.where(np.array(num_spikes) > 1)[0]

    if indy.size > 0 and not np.array_equal(unnorm_spikes, spikes):
        original_shift = [x / (unnorm_spikes[indy[0]][1] - unnorm_spikes[indy[0]][0]) * (spikes[indy[0]][1] - spikes[indy[0]][0]) for x in original_shift]

    original_shift = np.array(original_shift)
    original_shift -= np.mean(original_shift)

    if order == 1:
        spikes = spikes[::-1]
        original_shift = original_shift[::-1]
    elif order == 2:
        randi = np.random.permutation(num_trains)
        spikes = [spikes[x] for x in randi]
        original_shift = original_shift[randi]

    original_shift[np.abs(original_shift) < 1e-14] = 0

    if plotting == 1:
        fs = 15
        fig, ax = plt.subplots(figsize=(17, 10), dpi=80)
        plt.title("Rasterplot", color='k', fontsize=24)
        plt.xlabel('Time', color='k', fontsize=18)
        plt.ylabel('Spike Trains', color='k', fontsize=18)
        plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
        plt.xticks(np.arange(tmin,tmax+1,1000), fontsize=14)
        plt.yticks(np.arange(1,num_trains+1), np.arange(num_trains,0,-1),fontsize=14)
        plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1)
        plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
        plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
        for i in range(num_trains):
            for j in range(len(spikes[i])):
                plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-0.5), '-', color='k', linewidth=1)
        if num_trains < 12:
            plt.yticks(range(1, num_trains + 1), labels=range(num_trains, 0, -1))
        elif num_trains < 51:
            plt.yticks(range(num_trains, num_trains - 11, -10), labels=range(10, num_trains + 1, 10))
        else:
            plt.yticks(range(num_trains, num_trains - 21, -20), labels=range(20, num_trains + 1, 20))
        plt.box(True)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()
    spike_trains = []
    for i in spikes:
        spike_trains.append(spk.SpikeTrain(i, [tmin, tmax]))
    return [spike_trains, original_shift]