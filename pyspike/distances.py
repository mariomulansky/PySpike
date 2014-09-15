""" distances.py

Module containing several function to compute spike distances

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

import numpy as np

from pyspike import PieceWiseConstFunc

def spike_train_from_string(s, sep=' '):
    """ Converts a string of times into a SpikeTrain object.
    Params:
    - s: the string with (ordered) spike times
    - sep: The separator between the time numbers.
    Returns:
    - array of spike times
    """
    return np.fromstring(s, sep=sep)

def isi_distance(spikes1, spikes2, T_end, T_start=0.0):
    """ Computes the instantaneous isi-distance S_isi (t) of the two given spike
    trains. 
    Args:
    - spikes1, spikes2: ordered arrays of spike times.
    - T_end: end time of the observation interval.
    - T_start: begin of the observation interval (default=0.0).
    Returns:
    - PieceWiseConstFunc describing the isi-distance.
    """
    # add spikes at the beginning and end of the interval
    s1 = np.empty(len(spikes1)+2)
    s1[0] = T_start
    s1[-1] = T_end
    s1[1:-1] = spikes1
    s2 = np.empty(len(spikes2)+2)
    s2[0] = T_start
    s2[-1] = T_end
    s2[1:-1] = spikes2

    # compute the interspike interval
    nu1 = s1[1:]-s1[:-1]
    nu2 = s2[1:]-s2[:-1]
    
    # compute the isi-distance
    spike_events = np.empty(len(nu1)+len(nu2))
    spike_events[0] = T_start
    spike_events[-1] = T_end
    # the values have one entry less - the number of intervals between events
    isi_values = np.empty(len(spike_events)-1)
    # add the distance of the first events
    # isi_values[0] = nu1[0]/nu2[0] - 1.0 if nu1[0] <= nu2[0] \
    #                 else 1.0 - nu2[0]/nu1[0]
    isi_values[0] = (nu1[0]-nu2[0])/max(nu1[0],nu2[0])
    index1 = 0
    index2 = 0
    index = 1
    while True:
        # check which spike is next - from s1 or s2
        if s1[index1+1] <= s2[index2+1]:
            index1 += 1
            if index1 >= len(nu1):
                break
            spike_events[index] = s1[index1]
        else:
            index2 += 1
            if index2 >= len(nu2):
                break
            spike_events[index] = s2[index2]
        # compute the corresponding isi-distance
        isi_values[index] = (nu1[index1]-nu2[index2]) / \
                            max(nu1[index1], nu2[index2])
        index += 1
    return PieceWiseConstFunc(spike_events, isi_values)
