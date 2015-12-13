# Module containing several functions to compute the ISI profiles and distances
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import pyspike
from pyspike import PieceWiseConstFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_multi, \
    _generic_distance_matrix


############################################################
# isi_profile
############################################################
def isi_profile(spike_train1, spike_train2):
    """ Computes the isi-distance profile :math:`I(t)` of the two given
    spike trains. Retruns the profile as a PieceWiseConstFunc object. The
    ISI-values are defined positive :math:`I(t)>=0`.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`.SpikeTrain`
    :returns: The isi-distance profile :math:`I(t)`
    :rtype: :class:`.PieceWiseConstFunc`

    """
    # check whether the spike trains are defined for the same interval
    assert spike_train1.t_start == spike_train2.t_start, \
        "Given spike trains are not defined on the same interval!"
    assert spike_train1.t_end == spike_train2.t_end, \
        "Given spike trains are not defined on the same interval!"

    # load cython implementation
    try:
        from .cython.cython_profiles import isi_profile_cython \
            as isi_profile_impl
    except ImportError:
        if not(pyspike.disable_backend_warning):
            print("Warning: isi_profile_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from .cython.python_backend import isi_distance_python \
            as isi_profile_impl

    times, values = isi_profile_impl(spike_train1.get_spikes_non_empty(),
                                     spike_train2.get_spikes_non_empty(),
                                     spike_train1.t_start, spike_train1.t_end)
    return PieceWiseConstFunc(times, values)


############################################################
# isi_distance
############################################################
def isi_distance(spike_train1, spike_train2, interval=None):
    """ Computes the ISI-distance :math:`D_I` of the given spike trains. The
    isi-distance is the integral over the isi distance profile
    :math:`I(t)`:

    .. math:: D_I = \\int_{T_0}^{T_1} I(t) dt.

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

    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_distances import isi_distance_cython \
                as isi_distance_impl

            return isi_distance_impl(spike_train1.get_spikes_non_empty(),
                                     spike_train2.get_spikes_non_empty(),
                                     spike_train1.t_start, spike_train1.t_end)
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            return isi_profile(spike_train1, spike_train2).avrg(interval)
    else:
        # some specific interval is provided: use profile
        return isi_profile(spike_train1, spike_train2).avrg(interval)


############################################################
# isi_profile_multi
############################################################
def isi_profile_multi(spike_trains, indices=None):
    """ computes the multi-variate isi distance profile for a set of spike
    trains. That is the average isi-distance of all pairs of spike-trains:

    .. math:: <I(t)> = \\frac{2}{N(N-1)} \\sum_{<i,j>} I^{i,j},

    where the sum goes over all pairs <i,j>

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type state: list or None
    :returns: The averaged isi profile :math:`<I(t)>`
    :rtype: :class:`.PieceWiseConstFunc`
    """
    average_dist, M = _generic_profile_multi(spike_trains, isi_profile,
                                             indices)
    average_dist.mul_scalar(1.0/M)  # normalize
    return average_dist


############################################################
# isi_distance_multi
############################################################
def isi_distance_multi(spike_trains, indices=None, interval=None):
    """ computes the multi-variate isi-distance for a set of spike-trains.
    That is the time average of the multi-variate spike profile:

    .. math:: D_I = \\int_0^T \\frac{2}{N(N-1)} \\sum_{<i,j>} I^{i,j},

    where the sum goes over all pairs <i,j>

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :returns: The time-averaged multivariate ISI distance :math:`D_I`
    :rtype: double
    """
    return _generic_distance_multi(spike_trains, isi_distance, indices,
                                   interval)


############################################################
# isi_distance_matrix
############################################################
def isi_distance_matrix(spike_trains, indices=None, interval=None):
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
    return _generic_distance_matrix(spike_trains, isi_distance,
                                    indices, interval)
