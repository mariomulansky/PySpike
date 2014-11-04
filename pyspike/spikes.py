""" spikes.py

Module containing several function to load and transform spike trains

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

import numpy as np


############################################################
# add_auxiliary_spikes
############################################################
def add_auxiliary_spikes(spike_train, time_interval):
    """ Adds spikes at the beginning and end of the given time interval.

    :param spike_train: ordered array of spike times
    :param time_interval: A pair (T_start, T_end) of values representing the
                          start and end time of the spike train measurement or
                          a single value representing the end time, the T_start
                          is then assuemd as 0. Auxiliary spikes will be added
                          to the spike train at the beginning and end of this
                          interval, if they are not yet present.
    :type time_interval: pair of doubles or double
    :returns: spike train with additional spikes at T_start and T_end.

    """
    try:
        T_start = time_interval[0]
        T_end = time_interval[1]
    except:
        T_start = 0
        T_end = time_interval

    assert spike_train[0] >= T_start, \
        "Spike train has events before the given start time"
    assert spike_train[-1] <= T_end, \
        "Spike train has events after the given end time"
    if spike_train[0] != T_start:
        spike_train = np.insert(spike_train, 0, T_start)
    if spike_train[-1] != T_end:
        spike_train = np.append(spike_train, T_end)
    return spike_train


############################################################
# spike_train_from_string
############################################################
def spike_train_from_string(s, sep=' ', is_sorted=False):
    """ Converts a string of times into an array of spike times.

    :param s: the string with (ordered) spike times
    :param sep: The separator between the time numbers, default=' '.
    :param is_sorted: if True, the spike times are not sorted after loading,
    if False, spike times are sorted with `np.sort`
    :returns: array of spike times
    """
    if not(is_sorted):
        return np.sort(np.fromstring(s, sep=sep))
    else:
        return np.fromstring(s, sep=sep)


############################################################
# load_spike_trains_txt
############################################################
def load_spike_trains_from_txt(file_name, time_interval=None,
                               separator=' ', comment='#', sort=True):
    """ Loads a number of spike trains from a text file. Each line of the text
    file should contain one spike train as a sequence of spike times separated
    by `separator`. Empty lines as well as lines starting with `comment` are
    neglected. The `time_interval` represents the start and the end of the
    spike trains and it is used to add auxiliary spikes at the beginning and
    end of each spike train. However, if `time_interval == None`, no auxiliary
    spikes are added, but note that the Spike and ISI distance both require
    auxiliary spikes.

    :param file_name: The name of the text file.
    :param time_interval: A pair (T_start, T_end) of values representing the
                          start and end time of the spike train measurement
                          or a single value representing the end time, the
                          T_start is then assuemd as 0. Auxiliary spikes will
                          be added to the spike train at the beginning and end
                          of this interval.
    :param separator: The character used to seprate the values in the text file
    :param comment: Lines starting with this character are ignored.
    :param sort: If true, the spike times are order via `np.sort`, default=True
    :returns: list of spike trains
    """
    spike_trains = []
    spike_file = open(file_name, 'r')
    for line in spike_file:
        if len(line) > 1 and not line.startswith(comment):
            # use only the lines with actual data and not commented
            spike_train = spike_train_from_string(line, separator, sort)
            if time_interval is not None:  # add auxil. spikes if times given
                spike_train = add_auxiliary_spikes(spike_train, time_interval)
            spike_trains.append(spike_train)
    return spike_trains


############################################################
# merge_spike_trains
############################################################
def merge_spike_trains(spike_trains):
    """ Merges a number of spike trains into a single spike train.

    :param spike_trains: list of arrays of spike times
    :returns: spike train with the merged spike times
    """
    # get the lengths of the spike trains
    lens = np.array([len(st) for st in spike_trains])
    merged_spikes = np.empty(np.sum(lens))
    index = 0                             # the index for merged_spikes
    indices = np.zeros_like(lens)         # indices of the spike trains
    index_list = np.arange(len(indices))  # indices of indices of spike trains
                                          # that have not yet reached the end
    # list of the possible events in the spike trains
    vals = [spike_trains[i][indices[i]] for i in index_list]
    while len(index_list) > 0:
        i = np.argmin(vals)             # the next spike is the minimum
        merged_spikes[index] = vals[i]  # put it to the merged spike train
        i = index_list[i]
        index += 1                      # next index of merged spike train
        indices[i] += 1                 # next index for the chosen spike train
        if indices[i] >= lens[i]:       # remove spike train index if ended
            index_list = index_list[index_list != i]
        vals = [spike_trains[n][indices[n]] for n in index_list]
    return merged_spikes
