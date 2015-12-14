# Module containing several functions to compute SPIKE-Synchronization profiles
# and distances
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import numpy as np
from functools import partial
import pyspike
from pyspike import DiscreteFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_matrix


############################################################
# spike_sync_profile
############################################################
def spike_sync_profile(spike_train1, spike_train2, max_tau=None):
    """ Computes the spike-synchronization profile S_sync(t) of the two given
    spike trains. Returns the profile as a DiscreteFunction object. The S_sync
    values are either 1 or 0, indicating the presence or absence of a
    coincidence.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike-distance profile :math:`S_{sync}(t)`.
    :rtype: :class:`pyspike.function.DiscreteFunction`

    """
    # check whether the spike trains are defined for the same interval
    assert spike_train1.t_start == spike_train2.t_start, \
        "Given spike trains are not defined on the same interval!"
    assert spike_train1.t_end == spike_train2.t_end, \
        "Given spike trains are not defined on the same interval!"

    # cython implementation
    try:
        from .cython.cython_profiles import coincidence_profile_cython \
            as coincidence_profile_impl
    except ImportError:
        if not(pyspike.disable_backend_warning):
            print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from .cython.python_backend import coincidence_python \
            as coincidence_profile_impl

    if max_tau is None:
        max_tau = 0.0

    times, coincidences, multiplicity \
        = coincidence_profile_impl(spike_train1.spikes, spike_train2.spikes,
                                   spike_train1.t_start, spike_train1.t_end,
                                   max_tau)

    return DiscreteFunc(times, coincidences, multiplicity)


############################################################
# _spike_sync_values
############################################################
def _spike_sync_values(spike_train1, spike_train2, interval, max_tau):
    """" Internal function. Computes the summed coincidences and multiplicity
    for spike synchronization of the two given spike trains.

    Do not call this function directly, use `spike_sync` or `spike_sync_multi`
    instead.
    """
    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_distances import coincidence_value_cython \
                as coincidence_value_impl
            if max_tau is None:
                max_tau = 0.0
            c, mp = coincidence_value_impl(spike_train1.spikes,
                                           spike_train2.spikes,
                                           spike_train1.t_start,
                                           spike_train1.t_end,
                                           max_tau)
            return c, mp
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            return spike_sync_profile(spike_train1, spike_train2,
                                      max_tau).integral(interval)
    else:
        # some specific interval is provided: use profile
        return spike_sync_profile(spike_train1, spike_train2,
                                  max_tau).integral(interval)


############################################################
# spike_sync
############################################################
def spike_sync(spike_train1, spike_train2, interval=None, max_tau=None):
    """ Computes the spike synchronization value SYNC of the given spike
    trains. The spike synchronization value is the computed as the total number
    of coincidences divided by the total number of spikes:

    .. math:: SYNC = \sum_n C_n / N.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param interval: averaging interval given as a pair of floats (T0, T1),
                     if `None` the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike synchronization value.
    :rtype: `double`

    """
    c, mp = _spike_sync_values(spike_train1, spike_train2, interval, max_tau)
    return 1.0*c/mp


############################################################
# spike_sync_profile_multi
############################################################
def spike_sync_profile_multi(spike_trains, indices=None, max_tau=None):
    """ Computes the multi-variate spike synchronization profile for a set of
    spike trains. For each spike in the set of spike trains, the multi-variate
    profile is defined as the number of coincidences divided by the number of
    spike trains pairs involving the spike train of containing this spike,
    which is the number of spike trains minus one (N-1).

    :param spike_trains: list of :class:`pyspike.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The multi-variate spike sync profile :math:`<S_{sync}>(t)`
    :rtype: :class:`pyspike.function.DiscreteFunction`

    """
    prof_func = partial(spike_sync_profile, max_tau=max_tau)
    average_prof, M = _generic_profile_multi(spike_trains, prof_func,
                                             indices)
    # average_dist.mul_scalar(1.0/M)  # no normalization here!
    return average_prof


############################################################
# spike_sync_multi
############################################################
def spike_sync_multi(spike_trains, indices=None, interval=None, max_tau=None):
    """ Computes the multi-variate spike synchronization value for a set of
    spike trains.

    :param spike_trains: list of :class:`pyspike.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The multi-variate spike synchronization value SYNC.
    :rtype: double

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

    coincidence = 0.0
    mp = 0.0
    for (i, j) in pairs:
        c, m = _spike_sync_values(spike_trains[i], spike_trains[j],
                                  interval, max_tau)
        coincidence += c
        mp += m

    return coincidence/mp


############################################################
# spike_sync_matrix
############################################################
def spike_sync_matrix(spike_trains, indices=None, interval=None, max_tau=None):
    """ Computes the overall spike-synchronization value of all pairs of
    spike-trains.

    :param spike_trains: list of :class:`pyspike.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: 2D array with the pair wise time spike synchronization values
              :math:`SYNC_{ij}`
    :rtype: np.array

    """
    dist_func = partial(spike_sync, max_tau=max_tau)
    return _generic_distance_matrix(spike_trains, dist_func,
                                    indices, interval)
