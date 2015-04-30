# Module containing several functions to compute SPIKE-Synchronization profiles
# and distances
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from functools import partial
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
        "Given spike trains seems not to have auxiliary spikes!"
    assert spike_train1.t_end == spike_train2.t_end, \
        "Given spike trains seems not to have auxiliary spikes!"

    # cython implementation
    try:
        from cython.cython_distance import coincidence_cython \
            as coincidence_impl
    except ImportError:
        print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from cython.python_backend import coincidence_python \
            as coincidence_impl

    if max_tau is None:
        max_tau = 0.0

    times, coincidences, multiplicity = coincidence_impl(spike_train1.spikes,
                                                         spike_train2.spikes,
                                                         spike_train1.t_start,
                                                         spike_train1.t_end,
                                                         max_tau)

    return DiscreteFunc(times, coincidences, multiplicity)


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
    return spike_sync_profile(spike_train1, spike_train2,
                              max_tau).avrg(interval)


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
    return spike_sync_profile_multi(spike_trains, indices,
                                    max_tau).avrg(interval)


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
