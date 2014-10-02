""" distances.py

Module containing several functions to compute spike distances

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

import numpy as np
import threading

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

    # cython implementation
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

    # cython implementation
    from cython_distance import spike_distance_cython

    times, y_starts, y_ends = spike_distance_cython(spikes1, spikes2)

    return PieceWiseLinFunc(times, y_starts, y_ends)


############################################################
# multi_distance
############################################################
def multi_distance(spike_trains, pair_distance_func, indices=None):
    """ Internal implementation detail, don't call this function directly,
    use isi_distance_multi or spike_distance_multi instead.

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
# multi_distance_par
############################################################
def multi_distance_par(spike_trains, pair_distance_func, indices=None):
    """ parallel implementation of the multi-distance. Not currently used as
    it does not improve the performance.
    """

    num_threads = 2

    lock = threading.Lock()
    def run(spike_trains, index_pairs, average_dist):
        (i,j) = index_pairs[0]
        # print(i,j)
        this_avrg = pair_distance_func(spike_trains[i], spike_trains[j])
        for (i,j) in index_pairs[1:]:
            # print(i,j)
            current_dist = pair_distance_func(spike_trains[i], spike_trains[j])
            this_avrg.add(current_dist)
        with lock:
            average_dist.add(this_avrg)    

    if indices==None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
            "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(i,j) for i in indices for j in indices[i+1:]]
    num_pairs = len(pairs)

    # start with first pair
    (i,j) = pairs[0]
    average_dist = pair_distance_func(spike_trains[i], spike_trains[j])
    # remove the one we already computed
    pairs = pairs[1:]
    # distribute the rest into num_threads pieces
    clustered_pairs = [ pairs[i::num_threads] for i in xrange(num_threads) ]

    threads = []
    for pairs in clustered_pairs:
        t = threading.Thread(target=run, args=(spike_trains, pairs, average_dist))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    average_dist.mul_scalar(1.0/num_pairs) # normalize
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
    - indices: list of indices defining which spike-trains to use, 
    if None all given spike trains are used (default=None)
    Returns:
    - A PieceWiseLinFunc representing the averaged spike distance S
    """
    return multi_distance(spike_trains, spike_distance, indices)


def isi_distance_matrix(spike_trains, indices=None):
    """ Computes the average isi-distance of all pairs of spike-trains.
    Args:
    - spike_trains: list of spike trains
    - indices: list of indices defining which spike-trains to use
    if None all given spike-trains are used (default=None)
    Return:
    - a 2D array of size len(indices)*len(indices) containing the average 
    pair-wise isi-distance
    """
    if indices==None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
            "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(i,j) for i in indices for j in indices[i+1:]]
    
    distance_matrix = np.zeros((len(indices), len(indices)))
    for i,j in pairs:
        d = isi_distance(spike_trains[i], spike_trains[j]).abs_avrg()
        distance_matrix[i,j] = d
        distance_matrix[j,i] = d
    return distance_matrix
