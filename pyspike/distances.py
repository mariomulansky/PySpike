""" distances.py

Module containing several functions to compute spike distances

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

import numpy as np

from pyspike import PieceWiseConstFunc, PieceWiseLinFunc


############################################################
# add_auxiliary_spikes
############################################################
def add_auxiliary_spikes( spike_train, T_end , T_start=0.0):
    """ Adds spikes at the beginning (T_start) and end (T_end) of the 
    observation interval.
    Args:
    - spike_train: ordered array of spike times
    - T_end: end time of the observation interval
    - T_start: start time of the observation interval (default 0.0)
    Returns:
    - spike train with additional spikes at T_start and T_end.
    """
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
# isi_distance
############################################################
def isi_distance(spikes1, spikes2):
    """ Computes the instantaneous isi-distance S_isi (t) of the two given 
    spike trains. The spike trains are expected to have auxiliary spikes at the
    beginning and end of the interval. Use the function add_auxiliary_spikes to
    add those spikes to the spike train.
    Args:
    - spikes1, spikes2: ordered arrays of spike times with auxiliary spikes.
    Returns:
    - PieceWiseConstFunc describing the isi-distance.
    """
    # check for auxiliary spikes - first and last spikes should be identical
    assert spikes1[0]==spikes2[0], \
        "Given spike trains seems not to have auxiliary spikes!"
    assert spikes1[-1]==spikes2[-1], \
        "Given spike trains seems not to have auxiliary spikes!"

    # compile and load cython implementation
    import pyximport
    pyximport.install(setup_args={'include_dirs': [np.get_include()]})
    from cython_distance import isi_distance_cython

    times, values = isi_distance_cython(spikes1, spikes2)
    return PieceWiseConstFunc(times, values)


############################################################
# spike_distance
############################################################
def spike_distance(spikes1, spikes2):
    """ Computes the instantaneous spike-distance S_spike (t) of the two given
    spike trains. The spike trains are expected to have auxiliary spikes at the
    beginning and end of the interval. Use the function add_auxiliary_spikes to
    add those spikes to the spike train.
    Args:
    - spikes1, spikes2: ordered arrays of spike times with auxiliary spikes.
    Returns:
    - PieceWiseLinFunc describing the spike-distance.
    """
    # check for auxiliary spikes - first and last spikes should be identical
    assert spikes1[0]==spikes2[0], \
        "Given spike trains seems not to have auxiliary spikes!"
    assert spikes1[-1]==spikes2[-1], \
        "Given spike trains seems not to have auxiliary spikes!"

    # compile and load cython implementation
    import pyximport
    pyximport.install(setup_args={'include_dirs': [np.get_include()]})
    from cython_distance import spike_distance_cython

    times, y_starts, y_ends = spike_distance_cython(spikes1, spikes2)

    return PieceWiseLinFunc(times, y_starts, y_ends)


############################################################
# multi_distance
############################################################
def multi_distance(spike_trains, pair_distance_func, indices=None):
    """ Internal implementation detail, use isi_distance_multi or 
    spike_distance_multi.

    Computes the multi-variate distance for a set of spike-trains using the
    pair_dist_func to compute pair-wise distances. That is it computes the 
    average distance of all pairs of spike-trains:
    S(t) = 2/((N(N-1)) sum_{<i,j>} S_{i,j}, 
    where the sum goes over all pairs <i,j>.
    Args:
    - spike_trains: list of spike trains
    - pair_distance_func: function computing the distance of two spike trains
    - indices: list of indices defining which spike trains to use, 
    if None all given spike trains are used (default=None)
    Returns:
    - The averaged multi-variate distance of all pairs
    """
    if indices==None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
            "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(i,j) for i in indices for j in indices[i+1:]]
    # start with first pair
    (i,j) = pairs[0]
    average_dist = pair_distance_func(spike_trains[i], spike_trains[j])
    for (i,j) in pairs[1:]:
        current_dist = pair_distance_func(spike_trains[i], spike_trains[j])
        average_dist.add(current_dist)      # add to the average
    average_dist.mul_scalar(1.0/len(pairs)) # normalize
    return average_dist


############################################################
# isi_distance_multi
############################################################
def isi_distance_multi(spike_trains, indices=None):
    """ computes the multi-variate isi-distance for a set of spike-trains. That
    is the average isi-distance of all pairs of spike-trains:
    S(t) = 2/((N(N-1)) sum_{<i,j>} S_{i,j}, 
    where the sum goes over all pairs <i,j>
    Args:
    - spike_trains: list of spike trains
    - indices: list of indices defining which spike trains to use, 
    if None all given spike trains are used (default=None)
    Returns:
    - A PieceWiseConstFunc representing the averaged isi distance S
    """
    return multi_distance(spike_trains, isi_distance, indices)


############################################################
# spike_distance_multi
############################################################
def spike_distance_multi(spike_trains, indices=None):
    """ computes the multi-variate spike-distance for a set of spike-trains. 
    That is the average spike-distance of all pairs of spike-trains:
    S(t) = 2/((N(N-1)) sum_{<i,j>} S_{i,j}, 
    where the sum goes over all pairs <i,j>
    Args:
    - spike_trains: list of spike trains
    - indices: list of indices defining which spike trains to use, 
    if None all given spike trains are used (default=None)
    Returns:
    - A PieceWiseLinFunc representing the averaged spike distance S
    """
    return multi_distance(spike_trains, spike_distance, indices)


############################################################
############################################################
# VANILLA PYTHON IMPLEMENTATIONS OF ISI AND SPIKE DISTANCE
############################################################
############################################################


############################################################
# isi_distance_python
############################################################
def isi_distance_python(s1, s2):
    """ Plain Python implementation of the isi distance.
    """
    # compute the interspike interval
    nu1 = s1[1:]-s1[:-1]
    nu2 = s2[1:]-s2[:-1]
    
    # compute the isi-distance
    spike_events = np.empty(len(nu1)+len(nu2))
    spike_events[0] = s1[0]
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
        if s1[index1+1] < s2[index2+1]:
            index1 += 1
            # break condition relies on existence of spikes at T_end
            if index1 >= len(nu1):
                break
            spike_events[index] = s1[index1]
        elif s1[index1+1] > s2[index2+1]:
            index2 += 1
            if index2 >= len(nu2):
                break
            spike_events[index] = s2[index2]
        else: # s1[index1+1] == s2[index2+1]
            index1 += 1
            index2 += 1
            if (index1 >= len(nu1)) or (index2 >= len(nu2)):
                break
            spike_events[index] = s1[index1]
        # compute the corresponding isi-distance
        isi_values[index] = (nu1[index1]-nu2[index2]) / \
                            max(nu1[index1], nu2[index2])
        index += 1
    # the last event is the interval end
    spike_events[index] = s1[-1]
    # use only the data added above 
    # could be less than original length due to equal spike times
    return PieceWiseConstFunc(spike_events[:index+1], isi_values[:index])


############################################################
# get_min_dist
############################################################
def get_min_dist(spike_time, spike_train, start_index=0):
    """ Returns the minimal distance |spike_time - spike_train[i]| 
    with i>=start_index.
    """
    d = abs(spike_time - spike_train[start_index])
    start_index += 1
    while start_index < len(spike_train):
        d_temp = abs(spike_time - spike_train[start_index])
        if d_temp > d:
            break
        else:
            d = d_temp
        start_index += 1
    return d


############################################################
# spike_distance_python
############################################################
def spike_distance_python(spikes1, spikes2):
    """ Computes the instantaneous spike-distance S_spike (t) of the two given
    spike trains. The spike trains are expected to have auxiliary spikes at the
    beginning and end of the interval. Use the function add_auxiliary_spikes to
    add those spikes to the spike train.
    Args:
    - spikes1, spikes2: ordered arrays of spike times with auxiliary spikes.
    Returns:
    - PieceWiseLinFunc describing the spike-distance.
    """
    # check for auxiliary spikes - first and last spikes should be identical
    assert spikes1[0]==spikes2[0], \
        "Given spike trains seems not to have auxiliary spikes!"
    assert spikes1[-1]==spikes2[-1], \
        "Given spike trains seems not to have auxiliary spikes!"
    # shorter variables
    t1 = spikes1
    t2 = spikes2

    spike_events = np.empty(len(t1)+len(t2)-2)
    spike_events[0] = t1[0]
    y_starts = np.empty(len(spike_events)-1)
    y_ends = np.empty(len(spike_events)-1)

    index1 = 0
    index2 = 0
    index = 1
    dt_p1 = 0.0
    dt_f1 = get_min_dist(t1[1], t2, 0)
    dt_p2 = 0.0
    dt_f2 = get_min_dist(t2[1], t1, 0)
    isi1 = max(t1[1]-t1[0], t1[2]-t1[1])
    isi2 = max(t2[1]-t2[0], t2[2]-t2[1])
    s1 = dt_f1*(t1[1]-t1[0])/isi1
    s2 = dt_f2*(t2[1]-t2[0])/isi2
    y_starts[0] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
    while True:
        # print(index, index1, index2)
        if t1[index1+1] < t2[index2+1]:
            index1 += 1
            # break condition relies on existence of spikes at T_end
            if index1+1 >= len(t1):
                break
            spike_events[index] = t1[index1]
            # first calculate the previous interval end value
            dt_p1 = dt_f1 # the previous time now was the following time before
            s1 = dt_p1
            s2 = (dt_p2*(t2[index2+1]-t1[index1]) + dt_f2*(t1[index1]-t2[index2])) / isi2
            y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
            # now the next interval start value
            dt_f1 = get_min_dist(t1[index1+1], t2, index2)
            isi1 = t1[index1+1]-t1[index1]
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
        elif t1[index1+1] > t2[index2+1]:
            index2 += 1
            if index2+1 >= len(t2):
                break
            spike_events[index] = t2[index2]
            # first calculate the previous interval end value
            dt_p2 = dt_f2 # the previous time now was the following time before
            s1 = (dt_p1*(t1[index1+1]-t2[index2]) + dt_f1*(t2[index2]-t1[index1])) / isi1
            s2 = dt_p2
            y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
            # now the next interval start value
            dt_f2 = get_min_dist(t2[index2+1], t1, index1)
            #s2 = dt_f2
            isi2 = t2[index2+1]-t2[index2]
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
        else: # t1[index1+1] == t2[index2+1] - generate only one event
            index1 += 1
            index2 += 1
            if (index1+1 >= len(t1)) or (index2+1 >= len(t2)):
                break
            assert dt_f2 == 0.0
            assert dt_f1 == 0.0
            spike_events[index] = t1[index1]
            y_ends[index-1] = 0.0
            y_starts[index] = 0.0
            dt_p1 = 0.0
            dt_p2 = 0.0
            dt_f1 = get_min_dist(t1[index1+1], t2, index2)
            dt_f2 = get_min_dist(t2[index2+1], t1, index1)
            isi1 = t1[index1+1]-t1[index1]
            isi2 = t2[index2+1]-t2[index2]
        index += 1
    # the last event is the interval end
    spike_events[index] = t1[-1]
    # the ending value of the last interval
    isi1 = max(t1[-1]-t1[-2], t1[-2]-t1[-3])
    isi2 = max(t2[-1]-t2[-2], t2[-2]-t2[-3])
    s1 = dt_p1*(t1[-1]-t1[-2])/isi1
    s2 = dt_p2*(t2[-1]-t2[-2])/isi2
    y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
    # use only the data added above 
    # could be less than original length due to equal spike times
    return PieceWiseLinFunc(spike_events[:index+1], 
                            y_starts[:index], y_ends[:index])
