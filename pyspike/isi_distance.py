"""

Module containing several functions to compute the ISI profiles and distances

Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from pyspike import PieceWiseConstFunc
from pyspike.generic import _generic_profile_multi, _generic_distance_matrix


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
        from cython.cython_distance import isi_distance_cython \
            as isi_distance_impl
    except ImportError:
        print("Warning: isi_distance_cython not found. Make sure that PySpike \
is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from cython.python_backend import isi_distance_python \
            as isi_distance_impl

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
