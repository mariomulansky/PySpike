"""

Generic functions to compute multi-variate profiles and distance matrices.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import division

import numpy as np


############################################################
# _generic_profile_multi
############################################################
def _generic_profile_multi(spike_trains, pair_distance_func, indices=None):
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

    def divide_and_conquer(pairs1, pairs2):
        """ recursive calls by splitting the two lists in half.
        """
        L1 = len(pairs1)
        if L1 > 1:
            dist_prof1 = divide_and_conquer(pairs1[:L1//2],
                                            pairs1[L1//2:])
        else:
            dist_prof1 = pair_distance_func(spike_trains[pairs1[0][0]],
                                            spike_trains[pairs1[0][1]])
        L2 = len(pairs2)
        if L2 > 1:
            dist_prof2 = divide_and_conquer(pairs2[:L2//2],
                                            pairs2[L2//2:])
        else:
            dist_prof2 = pair_distance_func(spike_trains[pairs2[0][0]],
                                            spike_trains[pairs2[0][1]])
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
                                       spike_trains[pairs[0][1]])

    return avrg_dist, L


############################################################
# _generic_distance_multi
############################################################
def _generic_distance_multi(spike_trains, pair_distance_func,
                            indices=None, interval=None):
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
        avrg_dist += pair_distance_func(spike_trains[i], spike_trains[j],
                                        interval)

    return avrg_dist/len(pairs)


############################################################
# generic_distance_matrix
############################################################
def _generic_distance_matrix(spike_trains, dist_function,
                             indices=None, interval=None):
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
                          interval)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
    return distance_matrix
