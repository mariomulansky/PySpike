"""

Module containing several functions to compute SPIKE-Synchronization profiles
and distances

Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from functools import partial
from pyspike import DiscreteFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_matrix


############################################################
# spike_sync_profile
############################################################
def spike_sync_profile(spikes1, spikes2):
    """ Computes the spike-synchronization profile S_sync(t) of the two given
    spike trains. Returns the profile as a DiscreteFunction object. The S_sync
    values are either 1 or 0, indicating the presence or absence of a
    coincidence. The spike trains are expected to have auxiliary spikes at the
    beginning and end of the interval. Use the function add_auxiliary_spikes to
    add those spikes to the spike train.

    :param spikes1: ordered array of spike times with auxiliary spikes.
    :param spikes2: ordered array of spike times with auxiliary spikes.
    :returns: The spike-distance profile :math:`S_{sync}(t)`.
    :rtype: :class:`pyspike.function.DiscreteFunction`

    """

    # cython implementation
    try:
        from cython_distance import coincidence_cython \
            as coincidence_impl
    except ImportError:
        print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from python_backend import coincidence_python \
            as coincidence_impl

    times, coincidences, multiplicity = coincidence_impl(spikes1, spikes2)

    return DiscreteFunc(times, coincidences, multiplicity)


############################################################
# spike_sync
############################################################
def spike_sync(spikes1, spikes2, interval=None):
    """ Computes the spike synchronization value SYNC of the given spike
    trains. The spike synchronization value is the computed as the total number
    of coincidences divided by the total number of spikes:

    .. math:: SYNC = \sum_n C_n / N.

    :param spikes1: ordered array of spike times with auxiliary spikes.
    :param spikes2: ordered array of spike times with auxiliary spikes.
    :param interval: averaging interval given as a pair of floats (T0, T1),
                     if None the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The spike synchronization value.
    :rtype: double
    """
    return spike_sync_profile(spikes1, spikes2).avrg(interval)


############################################################
# spike_sync_profile_multi
############################################################
def spike_sync_profile_multi(spike_trains, indices=None):
    """ Computes the multi-variate spike synchronization profile for a set of
    spike trains. For each spike in the set of spike trains, the multi-variate
    profile is defined as the number of coincidences divided by the number of
    spike trains pairs involving the spike train of containing this spike,
    which is the number of spike trains minus one (N-1).

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :returns: The multi-variate spike sync profile :math:`<S_{sync}>(t)`
    :rtype: :class:`pyspike.function.DiscreteFunction`

    """
    prof_func = partial(spike_sync_profile)
    average_dist, M = _generic_profile_multi(spike_trains, prof_func,
                                             indices)
    # average_dist.mul_scalar(1.0/M)  # no normalization here!
    return average_dist


############################################################
# spike_distance_multi
############################################################
def spike_sync_multi(spike_trains, indices=None, interval=None):
    """ Computes the multi-variate spike synchronization value for a set of
    spike trains.

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The multi-variate spike synchronization value SYNC.
    :rtype: double
    """
    return spike_sync_profile_multi(spike_trains, indices).avrg(interval)


############################################################
# spike_sync_matrix
############################################################
def spike_sync_matrix(spike_trains, indices=None, interval=None):
    """ Computes the overall spike-synchronization value of all pairs of
    spike-trains.

    :param spike_trains: list of spike trains
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: 2D array with the pair wise time spike synchronization values
              :math:`SYNC_{ij}`
    :rtype: np.array
    """
    return _generic_distance_matrix(spike_trains, spike_sync,
                                    indices, interval)
