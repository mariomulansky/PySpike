# Module containing functions to compute the SPIKE directionality and the
# spike train order profile
# Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import numpy as np
import pyspike
from pyspike import DiscreteFunc
from functools import partial
from pyspike.generic import _generic_profile_multi, resolve_keywords
from pyspike.isi_lengths import default_thresh
from pyspike.spikes import reconcile_spike_trains, reconcile_spike_trains_bi


############################################################
# spike_directionality_values
############################################################

def spike_directionality_values(*args, **kwargs):
    """ Computes the spike directionality value for each spike in
    each spike train. Returns a list containing an array of spike directionality
    values for every given spike train.

    Valid call structures::

      spike_directionality_values(st1, st2)       # returns the bi-variate profile
      spike_directionality_values(st1, st2, st3)  # multi-variate profile of 3
                                                   # spike trains

      spike_trains = [st1, st2, st3, st4]          # list of spike trains
      spike_directionality_values(spike_trains)    # profile of the list of spike trains
      spike_directionality_values(spike_trains, indices=[0, 1])  # use only the spike trains
                                                                  # given by the indices

    Additonal arguments: 
    :param max_tau: Upper bound for coincidence window (default=None).
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)

    :returns: The spike directionality values :math:`D^n_i` as a list of arrays.
    """
    if len(args) == 1:
        return _spike_directionality_values_impl(args[0], **kwargs)
    else:
        return _spike_directionality_values_impl(args, **kwargs)


def _spike_directionality_values_impl(spike_trains, indices=None,
                                       interval=None, max_tau=None, **kwargs):
    """ Computes the multi-variate spike directionality profile 
    of the given spike trains.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike-directionality values.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
    ## get the keywords:
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh(spike_trains)

    if interval is not None:
        raise NotImplementedError("Parameter `interval` not supported.")
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # list of arrays for resulting asymmetry values
    asymmetry_list = [np.zeros_like(spike_trains[n].spikes) for n in indices]
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    # cython implementation
    try:
        from .cython.cython_directionality import \
            spike_directionality_profiles_cython as profile_impl
    except ImportError:
        pyspike.NoCythonWarn()

        # use python backend
        from .cython.directionality_python_backend import \
            spike_directionality_profile_python as profile_impl

    if max_tau is None:
        max_tau = 0.0

    for i, j in pairs:
        d1, d2 = profile_impl(spike_trains[i].spikes, spike_trains[j].spikes,
                              spike_trains[i].t_start, spike_trains[i].t_end,
                              max_tau, MRTS)
        asymmetry_list[i] += d1
        asymmetry_list[j] += d2
    for a in asymmetry_list:
        a /= len(spike_trains)-1
    return asymmetry_list


############################################################
# spike_directionality
############################################################
def spike_directionality(spike_train1, spike_train2, normalize=True,
                         interval=None, max_tau=None, **kwargs):
    """ Computes the overall spike directionality of the first spike train with
    respect to the second spike train.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param normalize: Normalize by the number of spikes (multiplicity).
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike train order profile :math:`E(t)`.
    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])

    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_directionality import \
                spike_directionality_cython as spike_directionality_impl
            if max_tau is None:
                max_tau = 0.0
            d = spike_directionality_impl(spike_train1.spikes,
                                          spike_train2.spikes,
                                          spike_train1.t_start,
                                          spike_train1.t_end,
                                          max_tau, MRTS)
            c = len(spike_train1.spikes)
        except ImportError:
            pyspike.NoCythonWarn()

            # use profile.
            d1, x = spike_directionality_values([spike_train1, spike_train2],
                                                 interval=interval,
                                                 max_tau=max_tau,
                                                 MRTS=MRTS)
            d = np.sum(d1)
            c = len(spike_train1.spikes)
        if normalize:
            return 1.0*d/c
        else:
            return d
    else:
        # some specific interval is provided: not yet implemented
        raise NotImplementedError("Parameter `interval` not supported.")


############################################################
# spike_directionality_matrix
############################################################
def spike_directionality_matrix(spike_trains, normalize=True, indices=None,
                                interval=None, max_tau=None, **kwargs):
    """ Computes the spike directionality matrix for the given spike trains.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param normalize: Normalize by the number of spikes (multiplicity).
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike-directionality values.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh(spike_trains)
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    distance_matrix = np.zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = spike_directionality(spike_trains[i], spike_trains[j], normalize,
                                 interval, max_tau=max_tau, 
                                 MRTS=MRTS, RI=RI, Reconcile=False)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = -d
    return distance_matrix


############################################################
# spike_train_order_profile
############################################################
def spike_train_order_profile(*args, **kwargs):
    """ Computes the spike train order profile :math:`E(t)` of the given
    spike trains. Returns the profile as a DiscreteFunction object.

    Valid call structures::

      spike_train_order_profile(st1, st2)       # returns the bi-variate profile
      spike_train_order_profile(st1, st2, st3)  # multi-variate profile of 3
                                                # spike trains

      spike_trains = [st1, st2, st3, st4]       # list of spike trains
      spike_train_order_profile(spike_trains)   # profile of the list of spike trains
      spike_train_order_profile(spike_trains, indices=[0, 1])  # use only the spike trains
                                                               # given by the indices

    Additonal arguments: 
    :param max_tau: Upper bound for coincidence window, `default=None`.
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)

    :returns: The spike train order profile :math:`E(t)`
    :rtype: :class:`.DiscreteFunction`
    """
    if len(args) == 1:
        return spike_train_order_profile_multi(args[0], **kwargs)
    elif len(args) == 2:
        return spike_train_order_profile_bi(args[0], args[1], **kwargs)
    else:
        return spike_train_order_profile_multi(args, **kwargs)


############################################################
# spike_train_order_profile_bi
############################################################
def spike_train_order_profile_bi(spike_train1, spike_train2, 
                                 max_tau=None, **kwargs):
    """ Computes the spike train order profile P(t) of the two given
    spike trains. Returns the profile as a DiscreteFunction object.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike train order profile :math:`E(t)`.
    :rtype: :class:`pyspike.function.DiscreteFunction`
    """
    if kwargs.get('Reconcile', True):
        spike_train1, spike_train2 = reconcile_spike_trains_bi(spike_train1, spike_train2)
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])

    # check whether the spike trains are defined for the same interval
    assert spike_train1.t_start == spike_train2.t_start, \
        "Given spike trains are not defined on the same interval!"
    assert spike_train1.t_end == spike_train2.t_end, \
        "Given spike trains are not defined on the same interval!"

    # cython implementation
    try:
        from .cython.cython_directionality import \
            spike_train_order_profile_cython as \
            spike_train_order_profile_impl
    except ImportError:
        # raise NotImplementedError()
        pyspike.NoCythonWarn()

        # use python backend
        from .cython.directionality_python_backend import \
            spike_train_order_profile_python as spike_train_order_profile_impl

    if max_tau is None:
        max_tau = 0.0

    times, coincidences, multiplicity \
        = spike_train_order_profile_impl(spike_train1.spikes,
                                         spike_train2.spikes,
                                         spike_train1.t_start,
                                         spike_train1.t_end,
                                         max_tau, MRTS)

    return DiscreteFunc(times, coincidences, multiplicity)


############################################################
# spike_train_order_profile_multi
############################################################
def spike_train_order_profile_multi(spike_trains, indices=None,
                                    max_tau=None, **kwargs):
    """ Computes the multi-variate spike train order profile for a set of
    spike trains. For each spike in the set of spike trains, the multi-variate
    profile is defined as the sum of asymmetry values divided by the number of
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
    prof_func = partial(spike_train_order_profile_bi, max_tau=max_tau)
    average_prof, M = _generic_profile_multi(spike_trains, prof_func,
                                             indices, **kwargs)
    return average_prof



############################################################
# _spike_train_order_impl
############################################################
def _spike_train_order_impl(spike_train1, spike_train2,
                            interval=None, max_tau=None, **kwargs):
    """ Implementation of bi-variatae spike train order value (Synfire Indicator).

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike train order value (Synfire Indicator)
    """
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh([spike_train1, spike_train2])
    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_directionality import \
                spike_train_order_cython as spike_train_order_func
            if max_tau is None:
                max_tau = 0.0
            c, mp = spike_train_order_func(spike_train1.spikes,
                                           spike_train2.spikes,
                                           spike_train1.t_start,
                                           spike_train1.t_end,
                                           max_tau, MRTS)
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            c, mp = spike_train_order_profile(spike_train1, spike_train2,
                                              max_tau=max_tau,
                                              MRTS=MRTS).integral(interval)
        return c, mp
    else:
        # some specific interval is provided: not yet implemented
        raise NotImplementedError("Parameter `interval` not supported.")


############################################################
# spike_train_order
############################################################
def spike_train_order(*args, **kwargs):
    """ Computes the spike train order (Synfire Indicator) of the given
    spike trains.

    Valid call structures::

      spike_train_order(st1, st2, normalize=True)  # normalized bi-variate
                                                    # spike train order
      spike_train_order(st1, st2, st3)  # multi-variate result of 3 spike trains

      spike_trains = [st1, st2, st3, st4]       # list of spike trains
      spike_train_order(spike_trains)   # result for the list of spike trains
      spike_train_order(spike_trains, indices=[0, 1])  # use only the spike trains
                                                       # given by the indices

    Additonal arguments: 
     - `max_tau` Upper bound for coincidence window, `default=None`.
     - `normalize` Flag indicating if the reslut should be normalized by the
       number of spikes , default=`False`


    :returns: The spike train order value (Synfire Indicator)
    """
    if len(args) == 1:
        return spike_train_order_multi(args[0], **kwargs)
    elif len(args) == 2:
        return spike_train_order_bi(args[0], args[1], **kwargs)
    else:
        return spike_train_order_multi(args, **kwargs)


############################################################
# spike_train_order_bi
############################################################
def spike_train_order_bi(spike_train1, spike_train2, normalize=True,
                         interval=None, max_tau=None, **kwargs):
    """ Computes the overall spike train order value (Synfire Indicator)
    for two spike trains.

    :param spike_train1: First spike train.
    :type spike_train1: :class:`pyspike.SpikeTrain`
    :param spike_train2: Second spike train.
    :type spike_train2: :class:`pyspike.SpikeTrain`
    :param normalize: Normalize by the number of spikes (multiplicity).
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: The spike train order value (Synfire Indicator)
    """
    c, mp = _spike_train_order_impl(spike_train1, spike_train2, interval, max_tau, **kwargs)
    if normalize:
        return 1.0*c/mp
    else:
        return c

############################################################
# spike_train_order_multi
############################################################
def spike_train_order_multi(spike_trains, indices=None, normalize=True,
                            interval=None, max_tau=None, **kwargs):
    """ Computes the overall spike train order value (Synfire Indicator)
    for many spike trains.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :param normalize: Normalize by the number of spike (multiplicity).
    :param interval: averaging interval given as a pair of floats, if None
                     the average over the whole function is computed.
    :type interval: Pair of floats or None.
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound.
    :returns: Spike train order values (Synfire Indicator) F for the given spike trains.
    :rtype: double
    """
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        MRTS = default_thresh(spike_trains)
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    e_total = 0.0
    m_total = 0.0
    for (i, j) in pairs:
        e, m = _spike_train_order_impl(spike_trains[i], spike_trains[j],
                                       interval, max_tau, MRTS=MRTS, RI=RI)
        e_total += e
        m_total += m

    if m == 0.0:
        return 1.0
    else:
        return e_total/m_total



############################################################
# optimal_spike_train_sorting_from_matrix
############################################################
def _optimal_spike_train_sorting_from_matrix(D, full_output=False):
    """ Finds the best sorting via simulated annealing.
    Returns the optimal permutation p and A value.
    Not for direct use, call :func:`.optimal_spike_train_sorting` instead.

    :param D: The directionality (Spike-ORDER) matrix.
    :param full_output: If true, then function will additionally return the
                        number of performed iterations (default=False)
    :return: (p, F) - tuple with the optimal permutation and synfire indicator.
             if `full_output=True` , (p, F, iter) is returned.
    """
    N = len(D)
    A = np.sum(np.triu(D, 0))

    p = np.arange(N)

    T_start = 2*np.max(D)    # starting temperature
    T_end = 1E-5 * T_start   # final temperature
    alpha = 0.9              # cooling factor

    try:
        from .cython.cython_simulated_annealing import sim_ann_cython as sim_ann
    except ImportError:
        raise NotImplementedError("PySpike with Cython required for computing spike train"
                                  " sorting!")

    p, A, total_iter = sim_ann(D, T_start, T_end, alpha)

    if full_output:
        return p, A, total_iter
    else:
        return p, A


############################################################
# optimal_spike_train_sorting
############################################################
def optimal_spike_train_sorting(spike_trains,  indices=None, interval=None,
                                max_tau=None, full_output=False, **kwargs):
    """ Finds the best sorting of the given spike trains by computing the spike
    directionality matrix and optimize the order using simulated annealing.
    For a detailed description of the algorithm see:
    `http://iopscience.iop.org/article/10.1088/1367-2630/aa68c3/meta`
    
    :param spike_trains: list of :class:`.SpikeTrain`
    :param indices: list of indices defining which spike trains to use,
                    if None all given spike trains are used (default=None)
    :type indices: list or None
    :param interval: time interval filter given as a pair of floats, if None
                     the full spike trains are used (default=None).
    :type interval: Pair of floats or None.
    :param max_tau: Maximum coincidence window size. If 0 or `None`, the
                    coincidence window has no upper bound (default=None).
    :param full_output: If true, then function will additionally return the
                        number of performed iterations (default=False)
    :return: (p, F) - tuple with the optimal permutation and synfire indicator.
             if `full_output=True` , (p, F, iter) is returned.
    """
    D = spike_directionality_matrix(spike_trains, normalize=False,
                                    indices=indices, interval=interval,
                                    max_tau=max_tau, **kwargs)
    return _optimal_spike_train_sorting_from_matrix(D, full_output)

############################################################
# permutate_matrix
############################################################
def permutate_matrix(D, p):
    """ Helper function that applies the permutation p to the columns and rows
    of matrix D. Return the permutated matrix :math:`D'[n,m] = D[p[n], p[m]]`.

    :param D: The matrix.
    :param d: The permutation.
    :return: The permuated matrix D', ie :math:`D'[n,m] = D[p[n], p[m]]`
    """
    N = len(D)
    D_p = np.empty_like(D)
    for n in range(N):
        for m in range(N):
            D_p[n, m] = D[p[n], p[m]]
    return D_p
