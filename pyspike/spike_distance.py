# Module containing several functions to compute SPIKE profiles and distances
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import pyspike
from pyspike import PieceWiseLinFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_multi, \
    _generic_distance_matrix, resolve_keywords
from pyspike.isi_lengths import default_thresh
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi


############################################################
# spike_profile
############################################################
def spike_profile(*args, **kwargs):
    """ Computes the spike-distance profile :math:`S(t)` of the given
    spike trains. Returns the profile as a PieceWiseConstLin object. The
    SPIKE-values are defined positive :math:`S(t)>=0`.

    Valid call structures::

      spike_profile(st1, st2)  # returns the bi-variate profile
      spike_profile(st1, st2, st3)  # multi-variate profile of 3 spike trains

      spike_trains = [st1, st2, st3, st4]  # list of spike trains
      spike_profile(spike_trains)  # profile of the list of spike trains
      spike_profile(spike_trains, indices=[0, 1])  # use only the spike trains
                                                   # given by the indices

    The multivariate spike-distance profile is defined as the average of all
    pairs of spike-trains:

    .. math:: <S(t)> = \\frac{2}{N(N-1)} \\sum_{<i,j>} S^{i, j}`,

    where the sum goes over all pairs <i,j>

    :returns: The spike-distance profile :math:`S(t)`
    :rtype: :class:`.PieceWiseConstLin`
    """
    if len(args) == 1:
        return spike_profile_multi(args[0], **kwargs)
    elif len(args) == 2:
        return spike_profile_bi(args[0], args[1], **kwargs)
    else:
        return spike_profile_multi(args, **kwargs)


############################################################
# spike_profile_bi
############################################################
def spike_profile_bi(spike_train1, spike_train2, **kwargs):
    """ Specific function to compute a bivariate SPIKE-profile. This is a
    deprecated function and should not be called directly. Use
    :func:`.spike_profile` to compute SPIKE-profiles.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`.SpikeTrain`
    :returns: The spike-distance profile :math:`S(t)`.
    :rtype: :class:`.PieceWiseLinFunc`

    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
    MRTS, RI = resolve_keywords(**kwargs)

    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])

    # cython implementation
    try:
        from .cython.cython_profiles import spike_profile_cython \
            as spike_profile_impl
    except ImportError:
        pyspike.NoCythonWarn()

        # use python backend
        from .cython.python_backend import spike_distance_python \
            as spike_profile_impl

    times, y_starts, y_ends = spike_profile_impl(
        spike_train1.get_spikes_non_empty(),
        spike_train2.get_spikes_non_empty(),
        spike_train1.t_start, spike_train1.t_end,
        MRTS, RI)

    return PieceWiseLinFunc(times, y_starts, y_ends)


############################################################
# spike_profile_multi
############################################################
def spike_profile_multi(spike_trains, indices=None, **kwargs):
    """ Specific function to compute a multivariate SPIKE-profile. This is a
    deprecated function and should not be called directly. Use
    :func:`.spike_profile` to compute SPIKE-profiles.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :returns: The averaged spike profile :math:`<S>(t)`
    :rtype: :class:`.PieceWiseLinFunc`

    """
    average_dist, M = _generic_profile_multi(spike_trains, spike_profile_bi,
                                             indices, **kwargs)
    average_dist.mul_scalar(1.0/M)  # normalize
    return average_dist


############################################################
# spike_distance
############################################################
def spike_distance(*args, **kwargs):
    """ Computes the SPIKE-distance :math:`D_S` of the given spike trains. The
    spike-distance is the integral over the spike distance profile
    :math:`D(t)`:

    .. math:: D_S = \\int_{T_0}^{T_1} S(t) dt.


    Valid call structures::

      spike_distance(st1, st2)  # returns the bi-variate distance
      spike_distance(st1, st2, st3)  # multi-variate distance of 3 spike trains

      spike_trains = [st1, st2, st3, st4]  # list of spike trains
      spike_distance(spike_trains)  # distance of the list of spike trains
      spike_distance(spike_trains, indices=[0, 1])  # use only the spike trains
                                                    # given by the indices

    In the multivariate case, the spike distance is given as the integral over
    the multivariate profile, that is the average profile of all spike train
    pairs:

    .. math::  D_S = \\int_0^T \\frac{2}{N(N-1)} \\sum_{<i,j>}
               S^{i, j} dt

    :returns: The spike-distance :math:`D_S`.
    :rtype: double
    """

    if len(args) == 1:
        return spike_distance_multi(args[0], **kwargs)
    elif len(args) == 2:
        return spike_distance_bi(args[0], args[1], **kwargs)
    else:
        return spike_distance_multi(args, **kwargs)


############################################################
# spike_distance_bi
############################################################
def spike_distance_bi(spike_train1, spike_train2, interval=None, **kwargs):
    """ Specific function to compute a bivariate SPIKE-distance. This is a
    deprecated function and should not be called directly. Use
    :func:`.spike_distance` to compute SPIKE-distances.

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
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
        kwargs['Reconcile'] = False
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])

    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_distances import spike_distance_cython \
                as spike_distance_impl
            return spike_distance_impl(spike_train1.get_spikes_non_empty(),
                                       spike_train2.get_spikes_non_empty(),
                                       spike_train1.t_start,
                                       spike_train1.t_end,
                                       MRTS, RI)
        except ImportError:
            # Cython backend not available: fall back to average profile
            return spike_profile_bi(spike_train1, spike_train2, 
                                    **kwargs).avrg(interval)
    else:
        # some specific interval is provided: compute the whole profile
        return spike_profile_bi(spike_train1, spike_train2, 
                                **kwargs).avrg(interval)


############################################################
# spike_distance_multi
############################################################
def spike_distance_multi(spike_trains, indices=None, interval=None, **kwargs):
    """ Specific function to compute a multivariate SPIKE-distance. This is a
    deprecated function and should not be called directly. Use
    :func:`.spike_distance` to compute SPIKE-distances.

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
    return _generic_distance_multi(spike_trains, spike_distance_bi, indices,
                                   interval, **kwargs)


############################################################
# spike_distance_matrix
############################################################
def spike_distance_matrix(spike_trains, indices=None, interval=None, **kwargs):
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
    return _generic_distance_matrix(spike_trains, spike_distance_bi,
                                    indices, interval, **kwargs)
