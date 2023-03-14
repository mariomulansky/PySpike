# Module containing several functions to compute SPIKE-Synchronization profiles
# and distances
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import numpy as np
from functools import partial
import pyspike
from pyspike import DiscreteFunc, SpikeTrain
from pyspike.generic import _generic_profile_multi, _generic_distance_matrix, resolve_keywords
from pyspike.isi_lengths import default_thresh
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi


############################################################
# spike_sync_profile
############################################################
def spike_sync_profile(*args, **kwargs):
    """ Computes the spike-synchronization profile S_sync(t) of the given
    spike trains. Returns the profile as a DiscreteFunction object. In the
    bivariate case, he S_sync values are either 1 or 0, indicating the presence
    or absence of a coincidence. For multi-variate cases, each spike in the set
    of spike trains, the profile is defined as the number of coincidences
    divided by the number of spike trains pairs involving the spike train of
    containing this spike, which is the number of spike trains minus one (N-1).

    Valid call structures::

      spike_sync_profile(st1, st2)  # returns the bi-variate profile
      spike_sync_profile(st1, st2, st3)  # multi-variate profile of 3 sts

      sts = [st1, st2, st3, st4]  # list of spike trains
      spike_sync_profile(sts)  # profile of the list of spike trains
      spike_sync_profile(sts, indices=[0, 1])  # use only the spike trains
                                               # given by the indices

    In the multivariate case, the profile is defined as the number of
    coincidences for each spike in the set of spike trains divided by the
    number of spike trains pairs involving the spike train of containing this
    spike, which is the number of spike trains minus one (N-1).

    :returns: The spike-sync profile :math:`S_{sync}(t)`.
    :rtype: :class:`pyspike.function.DiscreteFunction`
    """
    if len(args) == 1:
        return spike_sync_profile_multi(args[0], **kwargs)
    elif len(args) == 2:
        return spike_sync_profile_bi(args[0], args[1], **kwargs)
    else:
        return spike_sync_profile_multi(args, **kwargs)


############################################################
# spike_sync_profile_bi
############################################################
def spike_sync_profile_bi(spike_train1, spike_train2, max_tau=None, **kwargs):
    """ Specific function to compute a bivariate SPIKE-Sync-profile. This is a
    deprecated function and should not be called directly. Use
    :func:`.spike_sync_profile` to compute SPIKE-Sync-profiles.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike-sync profile :math:`S_{sync}(t)`.
    :rtype: :class:`pyspike.function.DiscreteFunction`

    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)

    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])

    # cython implementation
    try:
        from .cython.cython_profiles import coincidence_profile_cython \
            as coincidence_profile_impl
    except ImportError:
        pyspike.NoCythonWarn()

        # use python backend
        from .cython.python_backend import coincidence_python \
            as coincidence_profile_impl

    if max_tau is None:
        max_tau = 0.0

    times, coincidences, multiplicity \
        = coincidence_profile_impl(spike_train1.spikes, spike_train2.spikes,
                                   spike_train1.t_start, spike_train1.t_end,
                                   max_tau, MRTS)

    return DiscreteFunc(times, coincidences, multiplicity)


############################################################
# spike_sync_profile_multi
############################################################
def spike_sync_profile_multi(spike_trains, indices=None, max_tau=None, **kwargs):
    """  Specific function to compute a multivariate SPIKE-Sync-profile.
    This is a deprecated function and should not be called directly. Use
    :func:`.spike_sync_profile` to compute SPIKE-Sync-profiles.

    :param spike_trains: list of :class:`pyspike.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The multi-variate spike sync profile :math:`<S_{sync}>(t)`
    :rtype: :class:`pyspike.function.DiscreteFunction`

    """
    prof_func = partial(spike_sync_profile_bi, max_tau=max_tau)
    average_prof, M = _generic_profile_multi(spike_trains, prof_func,
                                             indices, **kwargs)
    # average_dist.mul_scalar(1.0/M)  # no normalization here!
    return average_prof


############################################################
# _spike_sync_values
############################################################
def _spike_sync_values(spike_train1, spike_train2, interval, max_tau, **kwargs):
    """" Internal function. Computes the summed coincidences and multiplicity
    for spike synchronization of the two given spike trains.

    Do not call this function directly, use `spike_sync` or `spike_sync_multi`
    instead.
    """
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])
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
                                           max_tau, MRTS)
            return c, mp
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            return spike_sync_profile_bi(spike_train1, spike_train2,
                                         max_tau, **kwargs).integral(interval)
    else:
        # some specific interval is provided: use profile
        return spike_sync_profile_bi(spike_train1, spike_train2,
                                     max_tau, **kwargs).integral(interval)


############################################################
# spike_sync
############################################################
def spike_sync(*args, **kwargs):
    """ Computes the spike synchronization value of the given spike
    trains. The spike synchronization value is the computed as the total number
    of coincidences divided by the total number of spikes:

    .. math:: SYNC = \sum_n C_n / N.


    Valid call structures::

      spike_sync(st1, st2)  # returns the bi-variate spike synchronization
      spike_sync(st1, st2, st3)  # multi-variate result for 3 spike trains

      spike_trains = [st1, st2, st3, st4]  # list of spike trains
      spike_sync(spike_trains)  # spike-sync of the list of spike trains
      spike_sync(spike_trains, indices=[0, 1])  # use only the spike trains
                                                # given by the indices

    The multivariate SPIKE-Sync is again defined as the overall ratio of all
    coincidence values divided by the total number of spikes.

    :returns: The spike synchronization value.
    :rtype: `double`
    """

    if len(args) == 1:
        return spike_sync_multi(args[0], **kwargs)
    elif len(args) == 2:
        return spike_sync_bi(args[0], args[1], **kwargs)
    else:
        return spike_sync_multi(args, **kwargs)


############################################################
# spike_sync_bi
############################################################
def spike_sync_bi(spike_train1, spike_train2, interval=None, max_tau=None, **kwargs):
    """ Specific function to compute a bivariate SPIKE-Sync value.
    This is a deprecated function and should not be called directly. Use
    :func:`.spike_sync` to compute SPIKE-Sync values.

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
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
        kwargs['Reconcile'] = False
    c, mp = _spike_sync_values(spike_train1, spike_train2, interval, max_tau, **kwargs)
    if mp == 0:
        return 1.0
    else:
        return 1.0*c/mp


############################################################
# spike_sync_multi
############################################################
def spike_sync_multi(spike_trains, indices=None, interval=None, max_tau=None, **kwargs):
    """ Specific function to compute a multivariate SPIKE-Sync value.
    This is a deprecated function and should not be called directly. Use
    :func:`.spike_sync` to compute SPIKE-Sync values.

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

    coincidence = 0.0
    mp = 0.0
    for (i, j) in pairs:
        c, m = _spike_sync_values(spike_trains[i], spike_trains[j],
                                  interval, max_tau, 
                                  **kwargs)
        coincidence += c
        mp += m

    if mp == 0.0:
        return 1.0
    else:
        return coincidence/mp


############################################################
# spike_sync_matrix
############################################################
def spike_sync_matrix(spike_trains, indices=None, interval=None, max_tau=None, **kwargs):
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
    dist_func = partial(spike_sync_bi, max_tau=max_tau)
    ShouldBeSync =  _generic_distance_matrix(spike_trains, dist_func,
                                    indices, interval, **kwargs)
    # These elements are not really distances, but spike-sync values
    #   The diagonal needs to reflect that:
    for i in range(ShouldBeSync.shape[0]):
        ShouldBeSync[i][i] = 1.0
    return ShouldBeSync


############################################################
# filter_by_spike_sync
############################################################
def filter_by_spike_sync(spike_trains, threshold, indices=None, max_tau=None,
                         return_removed_spikes=False, **kwargs):
    """ Removes the spikes with a multi-variate spike_sync value below
    threshold.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh(spike_trains)
    N = len(spike_trains)
    filtered_spike_trains = []
    removed_spike_trains = []

    # cython implementation
    try:
        from .cython.cython_profiles import coincidence_single_profile_cython \
            as coincidence_impl
    except ImportError:
        pyspike.NoCythonWarn()

        # use python backend
        from .cython.python_backend import coincidence_single_python \
            as coincidence_impl

    if max_tau is None:
        max_tau = 0.0

    for i, st in enumerate(spike_trains):
        coincidences = np.zeros_like(st)
        for j in range(N):
            if i == j:
                continue
            coincidences += coincidence_impl(st.spikes, spike_trains[j].spikes,
                                             st.t_start, st.t_end, max_tau, MRTS)
        filtered_spikes = st[coincidences > threshold*(N-1)]
        filtered_spike_trains.append(SpikeTrain(filtered_spikes,
                                                [st.t_start, st.t_end]))
        if return_removed_spikes:
            removed_spikes = st[coincidences <= threshold*(N-1)]
            removed_spike_trains.append(SpikeTrain(removed_spikes,
                                                   [st.t_start, st.t_end]))
    if return_removed_spikes:
        return [filtered_spike_trains, removed_spike_trains]
    else:
        return filtered_spike_trains
