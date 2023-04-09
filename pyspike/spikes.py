# Module containing several function to load and transform spike trains
# Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

import numpy as np
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