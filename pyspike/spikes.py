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


def merge_spike_trains( spike_trains ):
    """ Merges a number of spike trains into a single spike train.
    Params:
    - spike_trains: list of arrays of spike times
    Returns:
    - array with the merged spike times
    """
    # get the lengths of the spike trains
    lens = np.array([len(st) for st in spike_trains])
    merged_spikes = np.empty(np.sum(lens))
    index = 0
    indices = np.zeros_like(lens)
    vals = [spike_trains[i][indices[i]] for i in xrange(len(indices))]
    while len(indices) > 0:
        i = np.argmin(vals)
        merged_spikes[index] = vals[i]
        index += 1
        indices[i] += 1
        if indices[i] >= lens[i]:
            indices = np.delete(indices, i)
        vals = [spike_trains[i][indices[i]] for i in xrange(len(indices))]
    return merged_spikes
