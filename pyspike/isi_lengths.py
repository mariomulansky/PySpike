""" isi_lengths.py

Support for automatic threshold determination

Copyright 2023, Thomas Kreuz

Distributed under the BSD License
"""
import numpy as np

def isi_lengths(spike_times, t_start, t_end):
    """ Plain Python implementation of logic to extract ISI lengths
        In:  spike_times - spike times
             t_start, t_end - interval for ISI calculation
        Out: isi_lengths - ISI distance between consecutive elements of spike_events

        Note: the only complexities are with the edges and N==1
    """
    N = len(spike_times)
    if N == 0:
        return [t_end-t_start]

    if spike_times[0] > t_start:
        del_start = max(spike_times[0] - t_start, spike_times[1] - spike_times[0])\
                         if N > 1 else spike_times[0] - t_start
        i_start = 0
    else:
        del_start = spike_times[1] - spike_times[0]\
                         if N > 1 else t_start - spike_times[0]
        i_start = 1

    if spike_times[-1] < t_end:
        del_end = max(t_end - spike_times[-1], spike_times[-1] - spike_times[-2])\
                         if N > 1 else t_end - spike_times[0]
        i_end = N
    else:
        del_end = spike_times[-1] - spike_times[-2]\
                         if N > 1 else spike_times[0] - t_end
        i_end = N-1

    dels = [spike_times[i+1]-spike_times[i] for i in range(i_start, i_end-1)]

    isi_lengths = [del_start] + dels + [del_end]

    return isi_lengths

def default_thresh_(train_list, t_start, t_end):
    """ Implements default_thresh()
        In: train_list - list of list of spike times
            t_start, t_end - begin and end times for spikes
        Out: threshold
    """
    spike_pool = []
    for t in train_list:
        spike_pool += isi_lengths(t, t_start, t_end)

    spike_pool = np.array(spike_pool)
    sum_squares = np.sum(spike_pool * spike_pool)
    ss_avg = sum_squares/len(spike_pool)

    return np.sqrt(ss_avg)

def default_thresh(spike_train_list):
    """ Computes a default threshold for a list of spike trains
        In: spike_train_list - list of list of SpikeTrain object
        Out: threshold as specified in section 2.4 of
               "Measures of spike train synchrony for data with multiple time scales"
    """
    if len(spike_train_list) == 0:
        return 0
        
    st = spike_train_list[0]
    train_list = [st.spikes.tolist() for st in spike_train_list]

    return default_thresh_(train_list, st.t_start, st.t_end)