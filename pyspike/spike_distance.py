"""

Module containing several functions to compute SPIKE profiles and distances

Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from pyspike import PieceWiseLinFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_matrix


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
        from cython.cython_distance import spike_distance_cython \
            as spike_distance_impl
    except ImportError:
        print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from cython.python_backend import spike_distance_python \
            as spike_distance_impl

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
    average_dist, M = _generic_profile_multi(spike_trains, spike_profile,
                                             indices)
    average_dist.mul_scalar(1.0/M)  # normalize
    return average_dist


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
