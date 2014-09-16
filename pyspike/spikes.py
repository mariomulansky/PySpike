""" spikes.py

Module containing several function to load and transform spike trains

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

import numpy as np

def spike_train_from_string(s, sep=' '):
    """ Converts a string of times into an array of spike times.
    Params:
    - s: the string with (ordered) spike times
    - sep: The separator between the time numbers.
    Returns:
    - array of spike times
    """
    return np.fromstring(s, sep=sep)


def merge_spike_trains(spike_trains):
    """ Merges a number of spike trains into a single spike train.
    Params:
    - spike_trains: list of arrays of spike times
    Returns:
    - array with the merged spike times
    """
    # get the lengths of the spike trains
    lens = np.array([len(st) for st in spike_trains])
    merged_spikes = np.empty(np.sum(lens))
    index = 0                            # the index for merged_spikes
    indices = np.zeros_like(lens)        # indices of the spike trains
    index_list = np.arange(len(indices)) # indices of indices of spike trains
                                         # that have not yet reached the end
    # list of the possible events in the spike trains
    vals = [spike_trains[i][indices[i]] for i in index_list]
    while len(index_list) > 0:
        i = np.argmin(vals)              # the next spike is the minimum
        merged_spikes[index] = vals[i]   # put it to the merged spike train
        i = index_list[i]
        index += 1                       # next index of merged spike train
        indices[i] += 1                  # next index for the chosen spike train
        if indices[i] >= lens[i]:        # remove spike train index if ended
            index_list = index_list[index_list != i]
        vals = [spike_trains[i][indices[i]] for i in index_list]
    return merged_spikes
