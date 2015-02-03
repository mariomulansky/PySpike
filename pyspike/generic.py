"""

Generic functions to compute multi-variate profiles and distance matrices.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""


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
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]
    # start with first pair
    (i, j) = pairs[0]
    average_dist = pair_distance_func(spike_trains[i], spike_trains[j])
    for (i, j) in pairs[1:]:
        current_dist = pair_distance_func(spike_trains[i], spike_trains[j])
        average_dist.add(current_dist)       # add to the average
    return average_dist, len(pairs)


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
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    distance_matrix = np.zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = dist_function(spike_trains[i], spike_trains[j], interval)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
    return distance_matrix
