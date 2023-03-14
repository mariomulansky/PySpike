"""

Generic functions to compute multi-variate profiles and distance matrices.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import division
from pyspike.isi_lengths import default_thresh
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi
import numpy as np

def resolve_keywords(**kwargs):
    """ resolve keywords
        In: kwargs - dictionary of keywords
        out: MRTS - Minimum Relevant Time Scale, default 0.
             RI  - Rate Independent Adaptive distance, default False
    """
    if 'MRTS' in kwargs:
        MRTS = kwargs['MRTS']
    else:
        MRTS = 0.  # default
    if 'RI' in kwargs:
        RI = kwargs['RI']
    else:
        RI = False  # default
    return MRTS, RI


############################################################
# _generic_profile_multi
############################################################
def _generic_profile_multi(spike_trains, pair_distance_func, indices=None, **kwargs):
    """ Internal implementation detail, don't call this function directly,
    use isi_profile_multi or spike_profile_multi instead.

    Computes the multi-variate distance for a set of spike-trains using the
    pair_dist_func to compute pair-wise distances. That is it computes the
    average distance of all pairs of spike-trains:
    :math:`S(t) = 2/((N(N-1)) sum_{<i,j>} S_{i,j}`,
    where the sum goes over all pairs <i,j>.
    Args:
    - spike_trains: list of spike trains
    - pair_distance_func: function computing the distance of two spike trains
    - indices: list of indices defining which spike trains to use,
    if None all given spike trains are used (default=None)
    Returns:
    - The averaged multi-variate distance of all pairs
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
        kwargs['Reconcile'] = False

    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        kwargs['MRTS'] = default_thresh(spike_trains)

    def divide_and_conquer(pairs1, pairs2):
        """ recursive calls by splitting the two lists in half.
        """
        L1 = len(pairs1)
        if L1 > 1:
            dist_prof1 = divide_and_conquer(pairs1[:L1//2],
                                            pairs1[L1//2:])
        else:
            dist_prof1 = pair_distance_func(spike_trains[pairs1[0][0]],
                                            spike_trains[pairs1[0][1]],
                                            **kwargs)
        L2 = len(pairs2)
        if L2 > 1:
            dist_prof2 = divide_and_conquer(pairs2[:L2//2],
                                            pairs2[L2//2:])
        else:
            dist_prof2 = pair_distance_func(spike_trains[pairs2[0][0]],
                                            spike_trains[pairs2[0][1]], 
                                            **kwargs)
        dist_prof1.add(dist_prof2)
        return dist_prof1

    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    L = len(pairs)
    if L > 1:
        # recursive iteration through the list of pairs to get average profile
        avrg_dist = divide_and_conquer(pairs[:len(pairs)//2],
                                       pairs[len(pairs)//2:])
    else:
        avrg_dist = pair_distance_func(spike_trains[pairs[0][0]],
                                       spike_trains[pairs[0][1]], 
                                       **kwargs)

    return avrg_dist, L


############################################################
# _generic_distance_multi
############################################################
def _generic_distance_multi(spike_trains, pair_distance_func,
                            indices=None, interval=None, **kwargs):
    """ Internal implementation detail, don't call this function directly,
    use isi_distance_multi or spike_distance_multi instead.

    Computes the multi-variate distance for a set of spike-trains using the
    pair_dist_func to compute pair-wise distances. That is it computes the
    average distance of all pairs of spike-trains:
    :math:`S(t) = 2/((N(N-1)) sum_{<i,j>} D_{i,j}`,
    where the sum goes over all pairs <i,j>.
    Args:
    - spike_trains: list of spike trains
    - pair_distance_func: function computing the distance of two spike trains
    - indices: list of indices defining which spike trains to use,
    if None all given spike trains are used (default=None)
    Returns:
    - The averaged multi-variate distance of all pairs
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
        kwargs['Reconcile'] = False

    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        kwargs['MRTS'] = default_thresh(spike_trains)
    
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    avrg_dist = 0.0
    for (i, j) in pairs:
        one_dist = pair_distance_func(spike_trains[i], spike_trains[j],
                                        interval, **kwargs)
        avrg_dist += one_dist

    return avrg_dist/len(pairs)


############################################################
# generic_distance_matrix
############################################################
def _generic_distance_matrix(spike_trains, dist_function,
                             indices=None, interval=None, **kwargs):
    """ Internal implementation detail. Don't use this function directly.
    Instead use isi_distance_matrix or spike_distance_matrix.
    Computes the time averaged distance of all pairs of spike-trains.
    Args:
    - spike_trains: list of spike trains
    - indices: list of indices defining which spike-trains to use
    if None all given spike-trains are used (default=None)
    Return:
    - a 2D array of size len(indices)*len(indices) containing the average
    pair-wise distance
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
        kwargs['Reconcile'] = False
        
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        kwargs['MRTS'] = default_thresh(spike_trains)

    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(i, j) for i in range(len(indices))
             for j in range(i+1, len(indices))]

    distance_matrix = np.zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = dist_function(spike_trains[indices[i]], spike_trains[indices[j]],
                          interval, **kwargs)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
    return distance_matrix
