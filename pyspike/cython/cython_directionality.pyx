#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_directionality.pyx

cython implementation of the spike delay asymmetry measures

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_directionality.pyx

which gives::

  cython_directionality.html

"""

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from libc.math cimport fmax
from libc.math cimport fmin

from pyspike.cython.cython_get_tau cimport get_tau

#DTYPE = float
#ctypedef np.float_t DTYPE_t

############################################################
# spike_train_order_profile_cython
############################################################
def spike_train_order_profile_cython(double[:] spikes1, double[:] spikes2,
                                     double t_start, double t_end,
                                     double max_tau,
                                     double MRTS = 0.):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int i = -1
    cdef int j = -1
    cdef int n = 0
    cdef double[:] st = np.zeros(N1 + N2 + 2)  # spike times
    cdef double[:] a = np.zeros(N1 + N2 + 2)   # asymmetry values
    cdef double[:] mp = np.ones(N1 + N2 + 2)   # multiplicity
    cdef double interval = t_end - t_start
    cdef double tau

    cdef double true_max = t_end - t_start
    if max_tau > 0:
        true_max = fmin(true_max, 2*max_tau)

    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            st[n] = spikes1[i]
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # spike from spike train 1 after spike train 2
                # both get marked with -1
                a[n] = -1
                a[n-1] = -1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            st[n] = spikes2[j]
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # spike from spike train 1 before spike train 2
                # both get marked with 1
                a[n] = 1
                a[n-1] = 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            n += 1
            # add only one event with zero asymmetry value and multiplicity 2
            st[n] = spikes1[i]
            a[n] = 0
            mp[n] = 2

    st = st[:n+2]
    a = a[:n+2]
    mp = mp[:n+2]

    st[0] = t_start
    st[len(st)-1] = t_end
    if N1 + N2 > 0:
        a[0] = a[1]
        a[len(a)-1] = a[len(a)-2]
        mp[0] = mp[1]
        mp[len(mp)-1] = mp[len(mp)-2]
    else:
        a[0] = 1
        a[1] = 1

    return st, a, mp


############################################################
# spike_train_order_cython
############################################################
def spike_train_order_cython(double[:] spikes1, double[:] spikes2,
                             double t_start, double t_end, double max_tau,
                             double MRTS = 0.):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int i = -1
    cdef int j = -1
    cdef int d = 0
    cdef int mp = 0
    cdef double interval = t_end - t_start
    cdef double tau
    
    cdef double true_max = t_end - t_start
    if max_tau > 0:
        true_max = fmin(true_max, 2*max_tau)

    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            mp += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # spike in spike train 2 appeared before spike in spike train 1
                # mark with -1
                d -= 2
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            mp += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # spike in spike train 1 appeared before spike in spike train 2
                # mark with +1
                d += 2
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            # add only one event with multiplicity 2, but no asymmetry counting
            mp += 2

    if d == 0 and mp == 0:
        # empty spike trains -> spike sync = 1 by definition
        d = 1
        mp = 1

    return d, mp


############################################################
# spike_directionality_profiles_cython
############################################################
def spike_directionality_profiles_cython(double[:] spikes1,
                                         double[:] spikes2,
                                         double t_start, double t_end,
                                         double max_tau,
                                         double MRTS = 0.):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int i = -1
    cdef int j = -1
    cdef double[:] d1 = np.zeros(N1)  # directionality values
    cdef double[:] d2 = np.zeros(N2)  # directionality values
    cdef double interval = t_end - t_start
    cdef double tau

    cdef double true_max = t_end - t_start
    if max_tau > 0:
        true_max = fmin(true_max, 2*max_tau)

    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # spike from spike train 1 after spike train 2
                # leading spike gets +1, following spike -1
                d1[i] = -1
                d2[j] = +1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # spike from spike train 1 before spike train 2
                # leading spike gets +1, following spike -1
                d1[i] = +1
                d2[j] = -1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            # equal spike times: zero asymmetry value
            d1[i] = 0
            d2[j] = 0

    return d1, d2


############################################################
# spike_directionality_cython
############################################################
def spike_directionality_cython(double[:] spikes1,
                                double[:] spikes2,
                                double t_start, double t_end,
                                double max_tau,
                                double MRTS = 0.):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int i = -1
    cdef int j = -1
    cdef int d = 0  # directionality value
    cdef double interval = t_end - t_start
    cdef double tau

    cdef double true_max = t_end - t_start
    if max_tau > 0:
        true_max = fmin(true_max, 2*max_tau)

    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # spike from spike train 1 after spike train 2
                # leading spike gets +1, following spike -1
                d -= 1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # spike from spike train 1 before spike train 2
                # leading spike gets +1, following spike -1
                d += 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1

    return d
