# Module containing several functions to compute SPIKE profiles and distances
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import pyspike
from pyspike import PieceWiseLinFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_multi, \
    _generic_distance_matrix


############################################################
# spike_profile
############################################################
def spike_profile(spike_train1, spike_train2):
    """ Computes the spike-distance profile :math:`S(t)` of the two given spike
    trains. Returns the profile as a PieceWiseLinFunc object. The SPIKE-values
    are defined positive :math:`S(t)>=0`.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`.SpikeTrain`
    :returns: The spike-distance profile :math:`S(t)`.
    :rtype: :class:`.PieceWiseLinFunc`

    """
    # check whether the spike trains are defined for the same interval
    assert spike_train1.t_start == spike_train2.t_start, \
        "Given spike trains are not defined on the same interval!"
    assert spike_train1.t_end == spike_train2.t_end, \
        "Given spike trains are not defined on the same interval!"

    # cython implementation
    try:
        from .cython.cython_profiles import spike_profile_cython \
            as spike_profile_impl
    except ImportError:
        if not(pyspike.disable_backend_warning):
            print("Warning: spike_profile_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from .cython.python_backend import spike_distance_python \
            as spike_profile_impl

    times, y_starts, y_ends = spike_profile_impl(
        spike_train1.get_spikes_non_empty(),
        spike_train2.get_spikes_non_empty(),
        spike_train1.t_start, spike_train1.t_end)

    return PieceWiseLinFunc(times, y_starts, y_ends)


############################################################
# spike_distance
############################################################
def spike_distance(spike_train1, spike_train2, interval=None):
    """ Computes the spike-distance :math:`D_S` of the given spike trains. The
    spike-distance is the integral over the spike distance profile
    :math:`S(t)`:

    .. math:: D_S = \int_{T_0}^{T_1} S(t) dt.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`.SpikeTrain`
    :param interval: averaging interval given as a pair of floats (T0, T1),
                     if None the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The spike-distance.
    :rtype: double

    """
    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_distances import spike_distance_cython \
                as spike_distance_impl
            return spike_distance_impl(spike_train1.get_spikes_non_empty(),
                                       spike_train2.get_spikes_non_empty(),
                                       spike_train1.t_start,
                                       spike_train1.t_end)
        except ImportError:
            # Cython backend not available: fall back to average profile
            return spike_profile(spike_train1, spike_train2).avrg(interval)
    else:
        # some specific interval is provided: compute the whole profile
        return spike_profile(spike_train1, spike_train2).avrg(interval)


############################################################
# spike_profile_multi
############################################################
def spike_profile_multi(spike_trains, indices=None):
    """ Computes the multi-variate spike distance profile for a set of spike
    trains. That is the average spike-distance of all pairs of spike-trains:

    .. math:: <S(t)> = \\frac{2}{N(N-1)} \\sum_{<i,j>} S^{i, j}`,

    where the sum goes over all pairs <i,j>

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :returns: The averaged spike profile :math:`<S>(t)`
    :rtype: :class:`.PieceWiseLinFunc`

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

    .. math::  D_S = \\int_0^T \\frac{2}{N(N-1)} \\sum_{<i,j>}
               S^{i, j} dt

    where the sum goes over all pairs <i,j>

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The averaged multi-variate spike distance :math:`D_S`.
    :rtype: double
    """
    return _generic_distance_multi(spike_trains, spike_distance, indices,
                                   interval)


############################################################
# spike_distance_matrix
############################################################
def spike_distance_matrix(spike_trains, indices=None, interval=None):
    """ Computes the time averaged spike-distance of all pairs of spike-trains.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: 2D array with the pair wise time average spike distances
              :math:`D_S^{ij}`
    :rtype: np.array
    """
    return _generic_distance_matrix(spike_trains, spike_distance,
                                    indices, interval)
