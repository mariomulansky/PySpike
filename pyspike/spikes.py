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
# load_spike_trains_txt
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
    spike_file = open(file_name, 'r')
    for line in spike_file:
        if len(line) > 1 and not line.startswith(comment):
            # use only the lines with actual data and not commented
            spike_train = spike_train_from_string(line, edges,
                                                  separator, is_sorted)
            spike_trains.append(spike_train)
    return spike_trains


############################################################
# merge_spike_trains
############################################################
def merge_spike_trains(spike_trains):
    """ Merges a number of spike trains into a single spike train.

    :param spike_trains: list of :class:`.SpikeTrain`
    :returns: spike train with the merged spike times
    """
    # get the lengths of the spike trains
    lens = np.array([len(st.spikes) for st in spike_trains])
    merged_spikes = np.empty(np.sum(lens))
    index = 0                             # the index for merged_spikes
    indices = np.zeros_like(lens)         # indices of the spike trains
    index_list = np.arange(len(indices))  # indices of indices of spike trains
                                          # that have not yet reached the end
    # list of the possible events in the spike trains
    vals = [spike_trains[i].spikes[indices[i]] for i in index_list]
    while len(index_list) > 0:
        i = np.argmin(vals)             # the next spike is the minimum
        merged_spikes[index] = vals[i]  # put it to the merged spike train
        i = index_list[i]
        index += 1                      # next index of merged spike train
        indices[i] += 1                 # next index for the chosen spike train
        if indices[i] >= lens[i]:       # remove spike train index if ended
            index_list = index_list[index_list != i]
        vals = [spike_trains[n].spikes[indices[n]] for n in index_list]
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
