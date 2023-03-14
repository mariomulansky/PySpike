""" isi_distance.py
    Module containing several functions to compute the ISI profiles and distances
    Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
    Distributed under the BSD License
"""

from __future__ import absolute_import

import pyspike
from pyspike import PieceWiseConstFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_multi, \
    _generic_distance_matrix, resolve_keywords
from pyspike.isi_lengths import default_thresh
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi

############################################################
# isi_profile
############################################################
def isi_profile(*args, **kwargs):
    """ Computes the isi-distance profile :math:`I(t)` of the given
    spike trains. Returns the profile as a PieceWiseConstFunc object. The
    ISI-values are defined positive :math:`I(t)>=0`.

    Valid call structures::

      isi_profile(st1, st2)       # returns the bi-variate profile
      isi_profile(st1, st2, st3)  # multi-variate profile of 3 spike trains

      spike_trains = [st1, st2, st3, st4]  # list of spike trains
      isi_profile(spike_trains)   # profile of the list of spike trains
      isi_profile(spike_trains, indices=[0, 1])  # use only the spike trains
                                                 # given by the indices

    The multivariate ISI distance profile for a set of spike trains is defined
    as the average ISI-profile of all pairs of spike-trains:

    .. math:: <I(t)> = \\frac{2}{N(N-1)} \\sum_{<i,j>} I^{i,j},

    where the sum goes over all pairs <i,j>


    :returns: The isi-distance profile :math:`I(t)`
    :rtype: :class:`.PieceWiseConstFunc`
    """
    if len(args) == 1:
        return isi_profile_multi(args[0], **kwargs)
    elif len(args) == 2:
        return isi_profile_bi(args[0], args[1], **kwargs)
    else:
        return isi_profile_multi(args, **kwargs)


############################################################
# isi_profile_bi
############################################################
def isi_profile_bi(spike_train1, spike_train2, **kwargs):
    """ Specific function to compute a bivariate ISI-profile. This is a
    deprecated function and should not be called directly. Use
    :func:`.isi_profile` to compute ISI-profiles.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`.SpikeTrain`
    :returns: The isi-distance profile :math:`I(t)`
    :rtype: :class:`.PieceWiseConstFunc`

    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
        kwargs['Reconcile'] = False

    MRTS,RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])
        kwargs['MRTS'] = MRTS

    # load cython implementation
    try:
        from .cython.cython_profiles import isi_profile_cython \
            as isi_profile_impl
    except ImportError:
        pyspike.NoCythonWarn()

        # use python backend
        from .cython.python_backend import isi_distance_python \
            as isi_profile_impl

    times, values = isi_profile_impl(spike_train1.get_spikes_non_empty(),
                                     spike_train2.get_spikes_non_empty(),
                                     spike_train1.t_start, spike_train1.t_end,
                                     MRTS)
    return PieceWiseConstFunc(times, values)


############################################################
# isi_profile_multi
############################################################
def isi_profile_multi(spike_trains, indices=None, **kwargs):
    """ Specific function to compute the multivariate ISI-profile for a set of
    spike trains. This is a deprecated function and should not be called
    directly. Use :func:`.isi_profile` to compute ISI-profiles.


    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type state: list or None
    :returns: The averaged isi profile :math:`<I(t)>`
    :rtype: :class:`.PieceWiseConstFunc`
    """
    average_dist, M = _generic_profile_multi(spike_trains, isi_profile_bi,
                                             indices, **kwargs)
    average_dist.mul_scalar(1.0/M)  # normalize
    return average_dist


############################################################
# isi_distance
############################################################
def isi_distance(*args, **kwargs):
    """ Computes the ISI-distance :math:`D_I` of the given spike trains. The
    isi-distance is the integral over the isi distance profile
    :math:`I(t)`:

    .. math:: D_I = \\int_{T_0}^{T_1} I(t) dt.

    In the multivariate case it is the integral over the multivariate
    ISI-profile, i.e. the average profile over all spike train pairs:

    .. math:: D_I = \\int_0^T \\frac{2}{N(N-1)} \\sum_{<i,j>} I^{i,j},

    where the sum goes over all pairs <i,j>



    Valid call structures::

      isi_distance(st1, st2)  # returns the bi-variate distance
      isi_distance(st1, st2, st3)  # multi-variate distance of 3 spike trains

      spike_trains = [st1, st2, st3, st4]  # list of spike trains
      isi_distance(spike_trains)  # distance of the list of spike trains
      isi_distance(spike_trains, indices=[0, 1])  # use only the spike trains
                                                  # given by the indices

    :returns: The isi-distance :math:`D_I`.
    :rtype: double
    """

    if len(args) == 1:
        return isi_distance_multi(args[0], **kwargs)
    elif len(args) == 2:
        return isi_distance_bi(args[0], args[1], **kwargs)
    else:
        return isi_distance_multi(args, **kwargs)


############################################################
# _isi_distance_bi
############################################################
def isi_distance_bi(spike_train1, spike_train2, interval=None, **kwargs):
    """ Specific function to compute the bivariate ISI-distance.
    This is a deprecated function and should not be called directly. Use
    :func:`.isi_distance` to compute ISI-distances.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`.SpikeTrain`
    :param interval: averaging interval given as a pair of floats (T0, T1),
                     if None the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The isi-distance :math:`D_I`.
    :rtype: double
    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
        kwargs['Reconcile'] = False

    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])
        kwargs['MRTS'] = MRTS

    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_distances import isi_distance_cython \
                as isi_distance_impl

            return isi_distance_impl(spike_train1.get_spikes_non_empty(),
                                     spike_train2.get_spikes_non_empty(),
                                     spike_train1.t_start, spike_train1.t_end,
                                     MRTS)
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            return isi_profile_bi(spike_train1, spike_train2, **kwargs).avrg(interval)
    else:
        # some specific interval is provided: use profile
        return isi_profile_bi(spike_train1, spike_train2, **kwargs).avrg(interval)


############################################################
# isi_distance_multi
############################################################
def isi_distance_multi(spike_trains, indices=None, interval=None, **kwargs):
    """ Specific function to compute the multivariate ISI-distance.
    This is a deprecfated function and should not be called directly. Use
    :func:`.isi_distance` to compute ISI-distances.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The time-averaged multivariate ISI distance :math:`D_I`
    :rtype: double
    """
    return _generic_distance_multi(spike_trains, isi_distance_bi, indices,
                                   interval, **kwargs)


############################################################
# isi_distance_matrix
############################################################
def isi_distance_matrix(spike_trains, indices=None, interval=None, **kwargs):
    """ Computes the time averaged isi-distance of all pairs of spike-trains.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: 2D array with the pair wise time average isi distances
              :math:`D_{I}^{ij}`
    :rtype: np.array
    """
    return _generic_distance_matrix(spike_trains, isi_distance_bi,
                                    indices=indices, interval=interval,
                                    **kwargs)
