# Module containing functions to compute the SPIKE directionality and the
# spike train order profile
# Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import

import numpy as np
import pyspike
from pyspike import DiscreteFunc
from functools import partial
from pyspike.generic import _generic_profile_multi


############################################################
# spike_directionality
############################################################
def spike_directionality(spike_train1, spike_train2, normalize=True,
                         interval=None, max_tau=None):
    """ Computes the overall spike directionality for two spike trains.
    """
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
                                          max_tau)
            c = len(spike_train1.spikes)
        except ImportError:
            d1, x = spike_directionality_profiles([spike_train1, spike_train2],
                                                  interval=interval,
                                                  max_tau=max_tau)
            d = np.sum(d1)
            c = len(spike_train1.spikes)
        if normalize:
            return 1.0*d/c
        else:
            return d
    else:
        # some specific interval is provided: not yet implemented
        raise NotImplementedError()


############################################################
# spike_directionality_matrix
############################################################
def spike_directionality_matrix(spike_trains, normalize=True, indices=None,
                                interval=None, max_tau=None):
    """ Computes the spike directionaity matrix for the given spike trains.
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

    distance_matrix = np.zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = spike_directionality(spike_trains[i], spike_trains[j], normalize,
                                 interval, max_tau=max_tau)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = -d
    return distance_matrix


############################################################
# spike_directionality_profiles
############################################################
def spike_directionality_profiles(spike_trains, indices=None,
                                  interval=None, max_tau=None):
    """ Computes the spike directionality value for each spike in each spike
    train.
    """
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # list of arrays for reulting asymmetry values
    asymmetry_list = [np.zeros_like(st.spikes) for st in spike_trains]
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    # cython implementation
    try:
        from .cython.cython_directionality import \
            spike_directionality_profiles_cython as profile_impl
    except ImportError:
        if not(pyspike.disable_backend_warning):
            print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
        # use python backend
        from .cython.directionality_python_backend import \
            spike_directionality_profile_python as profile_impl

    if max_tau is None:
        max_tau = 0.0

    for i, j in pairs:
        d1, d2 = profile_impl(spike_trains[i].spikes, spike_trains[j].spikes,
                              spike_trains[i].t_start, spike_trains[i].t_end,
                              max_tau)
        asymmetry_list[i] += d1
        asymmetry_list[j] += d2
    for a in asymmetry_list:
        a /= len(spike_trains)-1
    return asymmetry_list


############################################################
# spike_train_order_profile
############################################################
def spike_train_order_profile(spike_train1, spike_train2, max_tau=None):
    """ Computes the spike train order profile P(t) of the two given
    spike trains. Returns the profile as a DiscreteFunction object.
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
        from .cython.cython_directionality import \
            spike_train_order_profile_cython as \
            spike_train_order_profile_impl
    except ImportError:
        # raise NotImplementedError()
        if not(pyspike.disable_backend_warning):
            print("Warning: spike_distance_cython not found. Make sure that \
PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
Falling back to slow python backend.")
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
                                         max_tau)

    return DiscreteFunc(times, coincidences, multiplicity)


############################################################
# spike_train_order
############################################################
def spike_train_order(spike_train1, spike_train2, normalize=True,
                      interval=None, max_tau=None):
    """ Computes the overall spike delay asymmetry value for two spike trains.
    """
    if interval is None:
        # distance over the whole interval is requested: use specific function
        # for optimal performance
        try:
            from .cython.cython_directionality import \
                spike_train_order_cython as spike_train_order_impl
            if max_tau is None:
                max_tau = 0.0
            c, mp = spike_train_order_impl(spike_train1.spikes,
                                           spike_train2.spikes,
                                           spike_train1.t_start,
                                           spike_train1.t_end,
                                           max_tau)
        except ImportError:
            # Cython backend not available: fall back to profile averaging
            c, mp = spike_train_order_profile(spike_train1, spike_train2,
                                              max_tau).integral(interval)
        if normalize:
            return 1.0*c/mp
        else:
            return c
    else:
        # some specific interval is provided: not yet implemented
        raise NotImplementedError()


############################################################
# spike_train_order_profile_multi
############################################################
def spike_train_order_profile_multi(spike_trains, indices=None,
                                    max_tau=None):
    """ Computes the multi-variate spike delay asymmetry profile for a set of
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
    prof_func = partial(spike_train_order_profile, max_tau=max_tau)
    average_prof, M = _generic_profile_multi(spike_trains, prof_func,
                                             indices)
    # average_dist.mul_scalar(1.0/M)  # no normalization here!
    return average_prof


############################################################
# optimal_spike_train_order_from_matrix
############################################################
def optimal_spike_train_order_from_matrix(D, full_output=False):
    """ finds the best sorting via simulated annealing.
    Returns the optimal permutation p and A value.
    Internal function, don't call directly! Use optimal_asymmetry_order
    instead.
    """
    N = len(D)
    A = np.sum(np.triu(D, 0))

    p = np.arange(N)

    T_start = 2*np.max(D)    # starting temperature
    T_end = 1E-5 * T_start   # final temperature
    alpha = 0.9              # cooling factor

    from .cython.cython_simulated_annealing import sim_ann_cython as sim_ann

    p, A, total_iter = sim_ann(D, T_start, T_end, alpha)

    # T = T_start
    # total_iter = 0
    # while T > T_end:
    #     iterations = 0
    #     succ_iter = 0
    #     # equilibrate for 100*N steps or 10*N successful steps
    #     while iterations < 100*N and succ_iter < 10*N:
    #         # exchange two rows and cols
    #         ind1 = np.random.randint(N-1)
    #         if ind1 < N-1:
    #             ind2 = ind1+1
    #         else:   # this can never happend
    #             ind2 = 0
    #         delta_A = -2*D[p[ind1], p[ind2]]
    #         if delta_A > 0.0 or exp(delta_A/T) > np.random.random():
    #             # swap indices
    #             p[ind1], p[ind2] = p[ind2], p[ind1]
    #             A += delta_A
    #             succ_iter += 1
    #         iterations += 1
    #     total_iter += iterations
    #     T *= alpha   # cool down
    #     if succ_iter == 0:
    #         break

    if full_output:
        return p, A, total_iter
    else:
        return p, A


############################################################
# optimal_spike_train_order
############################################################
def optimal_spike_train_order(spike_trains,  indices=None, interval=None,
                              max_tau=None, full_output=False):
    """ finds the best sorting of the given spike trains via simulated
    annealing.
    Returns the optimal permutation p and A value.
    """
    D = spike_directionality_matrix(spike_trains, normalize=False,
                                    indices=indices, interval=interval,
                                    max_tau=max_tau)
    return optimal_spike_train_order_from_matrix(D, full_output)


############################################################
# permutate_matrix
############################################################
def permutate_matrix(D, p):
    """ Applies the permutation p to the columns and rows of matrix D.
    Return the new permutated matrix.
    """
    N = len(D)
    D_p = np.empty_like(D)
    for n in range(N):
        for m in range(N):
            D_p[n, m] = D[p[n], p[m]]
    return D_p


# internal helper functions

############################################################
# _spike_directionality_profile
############################################################
# def _spike_directionality_profile(spike_train1, spike_train2,
#                                   max_tau=None):
#     """ Computes the spike delay asymmetry profile A(t) of the two given
#     spike trains. Returns the profile as a DiscreteFunction object.

#     :param spike_train1: First spike train.
#     :type spike_train1: :class:`pyspike.SpikeTrain`
#     :param spike_train2: Second spike train.
#     :type spike_train2: :class:`pyspike.SpikeTrain`
#     :param max_tau: Maximum coincidence window size. If 0 or `None`, the
#                     coincidence window has no upper bound.
#     :returns: The spike-distance profile :math:`S_{sync}(t)`.
#     :rtype: :class:`pyspike.function.DiscreteFunction`

#     """
#     # check whether the spike trains are defined for the same interval
#     assert spike_train1.t_start == spike_train2.t_start, \
#         "Given spike trains are not defined on the same interval!"
#     assert spike_train1.t_end == spike_train2.t_end, \
#         "Given spike trains are not defined on the same interval!"

#     # cython implementation
#     try:
#         from cython.cython_directionality import \
#             spike_train_order_profile_cython as \
#             spike_train_order_profile_impl
#     except ImportError:
#         # raise NotImplementedError()
#         if not(pyspike.disable_backend_warning):
#             print("Warning: spike_distance_cython not found. Make sure that \
# PySpike is installed by running\n 'python setup.py build_ext --inplace'!\n \
# Falling back to slow python backend.")
#         # use python backend
#         from cython.directionality_python_backend import \
#             spike_train_order_python as spike_train_order_profile_impl

#     if max_tau is None:
#         max_tau = 0.0

#     times, coincidences, multiplicity \
#         = spike_train_order_profile_impl(spike_train1.spikes,
#                                          spike_train2.spikes,
#                                          spike_train1.t_start,
#                                          spike_train1.t_end,
#                                          max_tau)

#     return DiscreteFunc(times, coincidences, multiplicity)
