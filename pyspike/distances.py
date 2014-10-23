""" distances.py

Module containing several functions to compute spike distances

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

import numpy as np
import threading

from pyspike import PieceWiseConstFunc, PieceWiseLinFunc


############################################################
# isi_profile
############################################################
def isi_profile(spikes1, spikes2):
    """ Computes the isi-distance profile :math:`S_{isi}(t)` of the two given
    spike trains. Retruns the profile as a PieceWiseConstFunc object. The S_isi
    values are defined positive S_isi(t)>=0.  The spike trains are expected
    to have auxiliary spikes at the beginning and end of the interval. Use the
    function add_auxiliary_spikes to add those spikes to the spike train.

    :param spikes1: ordered array of spike times with auxiliary spikes.
    :param spikes2: ordered array of spike times with auxiliary spikes.
    :returns: The isi-distance profile :math:`S_{isi}(t)`
    :rtype: :class:`pyspike.function.PieceWiseConstFunc`

    """
    # check for auxiliary spikes - first and last spikes should be identical
    assert spikes1[0] == spikes2[0], \
        "Given spike trains seems not to have auxiliary spikes!"
    assert spikes1[-1] == spikes2[-1], \
        "Given spike trains seems not to have auxiliary spikes!"

    # load cython implementation
    try:
        from cython_distance import isi_distance_cython as isi_distance_impl
    except ImportError:
        print("Warning: isi_distance_cython not found. Make sure that PySpike \
is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from python_backend import isi_distance_python as isi_distance_impl

    times, values = isi_distance_impl(spikes1, spikes2)
    return PieceWiseConstFunc(times, values)


############################################################
# isi_distance
############################################################
def isi_distance(spikes1, spikes2, interval=None):
    """ Computes the isi-distance I of the given spike trains. The
    isi-distance is the integral over the isi distance profile
    :math:`S_{isi}(t)`:

    .. math:: I = \int_{T_0}^{T_1} S_{isi}(t) dt.

    :param spikes1: ordered array of spike times with auxiliary spikes.
    :param spikes2: ordered array of spike times with auxiliary spikes.
    :param interval: averaging interval given as a pair of floats (T0, T1),
                     if None the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The isi-distance I.
    :rtype: double
    """
    return isi_profile(spikes1, spikes2).avrg(interval)


############################################################
# spike_profile
############################################################
def spike_profile(spikes1, spikes2):
    """ Computes the spike-distance profile S_spike(t) of the two given spike
    trains. Returns the profile as a PieceWiseLinFunc object. The S_spike
    values are defined positive S_spike(t)>=0. The spike trains are expected to
    have auxiliary spikes at the beginning and end of the interval. Use the
    function add_auxiliary_spikes to add those spikes to the spike train.

    :param spikes1: ordered array of spike times with auxiliary spikes.
    :param spikes2: ordered array of spike times with auxiliary spikes.
    :returns: The spike-distance profile :math:`S_{spike}(t)`.
    :rtype: :class:`pyspike.function.PieceWiseLinFunc`

    """
    # check for auxiliary spikes - first and last spikes should be identical
    assert spikes1[0] == spikes2[0], \
        "Given spike trains seems not to have auxiliary spikes!"
    assert spikes1[-1] == spikes2[-1], \
        "Given spike trains seems not to have auxiliary spikes!"

    # cython implementation
    try:
        from cython_distance import spike_distance_cython \
            as spike_distance_impl
    except ImportError:
        print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from python_backend import spike_distance_python as spike_distance_impl

    times, y_starts, y_ends = spike_distance_impl(spikes1, spikes2)
    return PieceWiseLinFunc(times, y_starts, y_ends)


############################################################
# spike_distance
############################################################
def spike_distance(spikes1, spikes2, interval=None):
    """ Computes the spike-distance S of the given spike trains. The
    spike-distance is the integral over the isi distance profile S_spike(t):

    .. math:: S = \int_{T_0}^{T_1} S_{spike}(t) dt.

    :param spikes1: ordered array of spike times with auxiliary spikes.
    :param spikes2: ordered array of spike times with auxiliary spikes.
    :param interval: averaging interval given as a pair of floats (T0, T1),
                     if None the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The spike-distance.
    :rtype: double

    """
    return spike_profile(spikes1, spikes2).avrg(interval)


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
    pairs = [(i, j) for i in indices for j in indices[i+1:]]
    # start with first pair
    (i, j) = pairs[0]
    average_dist = pair_distance_func(spike_trains[i], spike_trains[j])
    for (i, j) in pairs[1:]:
        current_dist = pair_distance_func(spike_trains[i], spike_trains[j])
        average_dist.add(current_dist)       # add to the average
    average_dist.mul_scalar(1.0/len(pairs))  # normalize
    return average_dist


############################################################
# multi_distance_par
############################################################
def _multi_distance_par(spike_trains, pair_distance_func, indices=None):
    """ parallel implementation of the multi-distance. Not currently used as
    it does not improve the performance.
    """

    num_threads = 2
    lock = threading.Lock()

    def run(spike_trains, index_pairs, average_dist):
        (i, j) = index_pairs[0]
        # print(i,j)
        this_avrg = pair_distance_func(spike_trains[i], spike_trains[j])
        for (i, j) in index_pairs[1:]:
            # print(i,j)
            current_dist = pair_distance_func(spike_trains[i], spike_trains[j])
            this_avrg.add(current_dist)
        with lock:
            average_dist.add(this_avrg)

    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(i, j) for i in indices for j in indices[i+1:]]
    num_pairs = len(pairs)

    # start with first pair
    (i, j) = pairs[0]
    average_dist = pair_distance_func(spike_trains[i], spike_trains[j])
    # remove the one we already computed
    pairs = pairs[1:]
    # distribute the rest into num_threads pieces
    clustered_pairs = [pairs[n::num_threads] for n in xrange(num_threads)]

    threads = []
    for pairs in clustered_pairs:
        t = threading.Thread(target=run, args=(spike_trains, pairs,
                                               average_dist))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    average_dist.mul_scalar(1.0/num_pairs)  # normalize
    return average_dist


############################################################
# isi_profile_multi
############################################################
def isi_profile_multi(spike_trains, indices=None):
    """ computes the multi-variate isi distance profile for a set of spike
    trains. That is the average isi-distance of all pairs of spike-trains:
    S_isi(t) = 2/((N(N-1)) sum_{<i,j>} S_{isi}^{i,j},
    where the sum goes over all pairs <i,j>

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type state: list or None
    :returns: The averaged isi profile :math:`<S_{isi}>(t)`
    :rtype: :class:`pyspike.function.PieceWiseConstFunc`
    """
    return _generic_profile_multi(spike_trains, isi_profile, indices)


############################################################
# isi_distance_multi
############################################################
def isi_distance_multi(spike_trains, indices=None, interval=None):
    """ computes the multi-variate isi-distance for a set of spike-trains.
    That is the time average of the multi-variate spike profile:
    I = \int_0^T 2/((N(N-1)) sum_{<i,j>} S_{isi}^{i,j},
    where the sum goes over all pairs <i,j>

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The time-averaged isi distance :math:`I`
    :rtype: double
    """
    return isi_profile_multi(spike_trains, indices).avrg(interval)


############################################################
# spike_profile_multi
############################################################
def spike_profile_multi(spike_trains, indices=None):
    """ Computes the multi-variate spike distance profile for a set of spike
    trains. That is the average spike-distance of all pairs of spike-trains:
    :math:`S_spike(t) = 2/((N(N-1)) sum_{<i,j>} S_{spike}^{i, j}`,
    where the sum goes over all pairs <i,j>

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :returns: The averaged spike profile :math:`<S_{spike}>(t)`
    :rtype: :class:`pyspike.function.PieceWiseLinFunc`

    """
    return _generic_profile_multi(spike_trains, spike_profile, indices)


############################################################
# spike_distance_multi
############################################################
def spike_distance_multi(spike_trains, indices=None, interval=None):
    """ Computes the multi-variate spike distance for a set of spike trains.
    That is the time average of the multi-variate spike profile:
    S_{spike} = \int_0^T 2/((N(N-1)) sum_{<i,j>} S_{spike}^{i, j} dt
    where the sum goes over all pairs <i,j>

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The averaged spike distance S.
    :rtype: double
    """
    return spike_profile_multi(spike_trains, indices).avrg(interval)


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
    pairs = [(i, j) for i in indices for j in indices[i+1:]]

    distance_matrix = np.zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = dist_function(spike_trains[i], spike_trains[j], interval)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
    return distance_matrix


############################################################
# isi_distance_matrix
############################################################
def isi_distance_matrix(spike_trains, indices=None, interval=None):
    """ Computes the time averaged isi-distance of all pairs of spike-trains.

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: 2D array with the pair wise time average isi distances
              :math:`I_{ij}`
    :rtype: np.array
    """
    return _generic_distance_matrix(spike_trains, isi_distance,
                                    indices, interval)


############################################################
# spike_distance_matrix
############################################################
def spike_distance_matrix(spike_trains, indices=None, interval=None):
    """ Computes the time averaged spike-distance of all pairs of spike-trains.

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: 2D array with the pair wise time average spike distances
              :math:`S_{ij}`
    :rtype: np.array
    """
    return _generic_distance_matrix(spike_trains, spike_distance,
                                    indices, interval)
