#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_distances.pyx

cython implementation of the isi- and spike-distance

Note: using cython memoryviews (e.g. double[:]) instead of ndarray objects
improves the performance of spike_distance by a factor of 10!

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_distance.pyx

which gives::

  cython_distance.html

"""

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from libc.math cimport fmax
from libc.math cimport fmin

DTYPE = np.float
ctypedef np.float_t DTYPE_t


############################################################
# isi_distance_cython
############################################################
def isi_distance_cython(double[:] s1, 
                        double[:] s2):

    cdef double[:] spike_events
    cdef double[:] isi_values
    cdef int index1, index2, index
    cdef int N1, N2
    cdef double nu1, nu2
    N1 = len(s1)-1
    N2 = len(s2)-1

    nu1 = s1[1]-s1[0]
    nu2 = s2[1]-s2[0]
    spike_events = np.empty(N1+N2)
    spike_events[0] = s1[0]
    # the values have one entry less - the number of intervals between events
    isi_values = np.empty(N1+N2-1)

    with nogil: # release the interpreter to allow multithreading
        isi_values[0] = fabs(nu1-nu2)/fmax(nu1, nu2)
        index1 = 0
        index2 = 0
        index = 1
        while True:
            # check which spike is next - from s1 or s2
            if s1[index1+1] < s2[index2+1]:
                index1 += 1
                # break condition relies on existence of spikes at T_end
                if index1 >= N1:
                    break
                spike_events[index] = s1[index1]
                nu1 = s1[index1+1]-s1[index1]
            elif s1[index1+1] > s2[index2+1]:
                index2 += 1
                if index2 >= N2:
                    break
                spike_events[index] = s2[index2]
                nu2 = s2[index2+1]-s2[index2]
            else: # s1[index1+1] == s2[index2+1]
                index1 += 1
                index2 += 1
                if (index1 >= N1) or (index2 >= N2):
                    break
                spike_events[index] = s1[index1]
                nu1 = s1[index1+1]-s1[index1]
                nu2 = s2[index2+1]-s2[index2]            
            # compute the corresponding isi-distance
            isi_values[index] = fabs(nu1 - nu2) / fmax(nu1, nu2)
            index += 1
        # the last event is the interval end
        spike_events[index] = s1[N1]
    # end nogil

    return spike_events[:index+1], isi_values[:index]


############################################################
# get_min_dist_cython
############################################################
cdef inline double get_min_dist_cython(double spike_time, 
                                       double[:] spike_train,
                                       # use memory view to ensure inlining
                                       # np.ndarray[DTYPE_t,ndim=1] spike_train,
                                       int N,
                                       int start_index=0) nogil:
    """ Returns the minimal distance |spike_time - spike_train[i]| 
    with i>=start_index.
    """
    cdef double d, d_temp
    d = fabs(spike_time - spike_train[start_index])
    start_index += 1
    while start_index < N:
        d_temp = fabs(spike_time - spike_train[start_index])
        if d_temp > d:
            break
        else:
            d = d_temp
        start_index += 1
    return d


############################################################
# isi_avrg_cython
############################################################
cdef inline double isi_avrg_cython(double isi1, double isi2) nogil:
    return 0.5*(isi1+isi2)*(isi1+isi2)
    # alternative definition to obtain <S> ~ 0.5 for Poisson spikes
    # return 0.5*(isi1*isi1+isi2*isi2)


############################################################
# spike_distance_cython
############################################################
def spike_distance_cython(double[:] t1, 
                          double[:] t2):

    cdef double[:] spike_events
    cdef double[:] y_starts
    cdef double[:] y_ends

    cdef int N1, N2, index1, index2, index
    cdef double dt_p1, dt_p2, dt_f1, dt_f2, isi1, isi2, s1, s2

    N1 = len(t1)
    N2 = len(t2)

    spike_events = np.empty(N1+N2-2)
    spike_events[0] = t1[0]
    y_starts = np.empty(len(spike_events)-1)
    y_ends = np.empty(len(spike_events)-1)

    with nogil: # release the interpreter to allow multithreading
        index1 = 0
        index2 = 0
        index = 1
        dt_p1 = 0.0
        dt_f1 = get_min_dist_cython(t1[1], t2, N2, 0)
        dt_p2 = 0.0
        dt_f2 = get_min_dist_cython(t2[1], t1, N1, 0)
        isi1 = max(t1[1]-t1[0], t1[2]-t1[1])
        isi2 = max(t2[1]-t2[0], t2[2]-t2[1])
        s1 = dt_f1*(t1[1]-t1[0])/isi1
        s2 = dt_f2*(t2[1]-t2[0])/isi2
        y_starts[0] = (s1*isi2 + s2*isi1) / isi_avrg_cython(isi1, isi2)
        while True:
            # print(index, index1, index2)
            if t1[index1+1] < t2[index2+1]:
                index1 += 1
                # break condition relies on existence of spikes at T_end
                if index1+1 >= N1:
                    break
                spike_events[index] = t1[index1]
                # first calculate the previous interval end value
                dt_p1 = dt_f1 # the previous time now was the following time before
                s1 = dt_p1
                s2 = (dt_p2*(t2[index2+1]-t1[index1]) + 
                      dt_f2*(t1[index1]-t2[index2])) / isi2
                y_ends[index-1] = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
                # now the next interval start value
                dt_f1 = get_min_dist_cython(t1[index1+1], t2, N2, index2)
                isi1 = t1[index1+1]-t1[index1]
                # s2 is the same as above, thus we can compute y2 immediately
                y_starts[index] = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
            elif t1[index1+1] > t2[index2+1]:
                index2 += 1
                if index2+1 >= N2:
                    break
                spike_events[index] = t2[index2]
                # first calculate the previous interval end value
                dt_p2 = dt_f2 # the previous time now was the following time before
                s1 = (dt_p1*(t1[index1+1]-t2[index2]) + 
                      dt_f1*(t2[index2]-t1[index1])) / isi1
                s2 = dt_p2
                y_ends[index-1] = (s1*isi2 + s2*isi1) / isi_avrg_cython(isi1, isi2)
                # now the next interval start value
                dt_f2 = get_min_dist_cython(t2[index2+1], t1, N1, index1)
                #s2 = dt_f2
                isi2 = t2[index2+1]-t2[index2]
                # s2 is the same as above, thus we can compute y2 immediately
                y_starts[index] = (s1*isi2 + s2*isi1)/isi_avrg_cython(isi1, isi2)
            else: # t1[index1+1] == t2[index2+1] - generate only one event
                index1 += 1
                index2 += 1
                if (index1+1 >= N1) or (index2+1 >= N2):
                    break
                spike_events[index] = t1[index1]
                y_ends[index-1] = 0.0
                y_starts[index] = 0.0
                dt_p1 = 0.0
                dt_p2 = 0.0
                dt_f1 = get_min_dist_cython(t1[index1+1], t2, N2, index2)
                dt_f2 = get_min_dist_cython(t2[index2+1], t1, N1, index1)
                isi1 = t1[index1+1]-t1[index1]
                isi2 = t2[index2+1]-t2[index2]
            index += 1
        # the last event is the interval end
        spike_events[index] = t1[N1-1]
        # the ending value of the last interval
        isi1 = max(t1[N1-1]-t1[N1-2], t1[N1-2]-t1[N1-3])
        isi2 = max(t2[N2-1]-t2[N2-2], t2[N2-2]-t2[N2-3])
        s1 = dt_p1*(t1[N1-1]-t1[N1-2])/isi1
        s2 = dt_p2*(t2[N2-1]-t2[N2-2])/isi2
        y_ends[index-1] = (s1*isi2 + s2*isi1) / isi_avrg_cython(isi1, isi2)
    # end nogil

    # use only the data added above 
    # could be less than original length due to equal spike times
    return spike_events[:index+1], y_starts[:index], y_ends[:index]



############################################################
# coincidence_python
############################################################
cdef inline double get_tau(double[:] spikes1, double[:] spikes2,
                           int i, int j, max_tau):
    cdef double m = 1E100   # some huge number
    cdef int N1 = len(spikes1)-2
    cdef int N2 = len(spikes2)-2
    if i < N1:
        m = fmin(m, spikes1[i+1]-spikes1[i])
    if j < N2:
        m = fmin(m, spikes2[j+1]-spikes2[j])
    if i > 1:
        m = fmin(m, spikes1[i]-spikes1[i-1])
    if j > 1:
        m = fmin(m, spikes2[j]-spikes2[j-1])
    m *= 0.5
    if max_tau > 0.0:
        m = fmin(m, max_tau)
    return m
    

############################################################
# coincidence_cython
############################################################
def coincidence_cython(double[:] spikes1, double[:] spikes2, double max_tau):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int i = 0
    cdef int j = 0
    cdef int n = 0
    cdef double[:] st = np.zeros(N1 + N2 - 2)  # spike times
    cdef double[:] c = np.zeros(N1 + N2 - 2)   # coincidences
    cdef double[:] mp = np.ones(N1 + N2 - 2)   # multiplicity
    cdef double tau
    while n < N1 + N2 - 2:
        if spikes1[i+1] < spikes2[j+1]:
            i += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            st[n] = spikes1[i]
            if j > 0 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # both get marked with 1
                c[n] = 1
                c[n-1] = 1
        elif spikes1[i+1] > spikes2[j+1]:
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            st[n] = spikes2[j]
            if i > 0 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # both get marked with 1
                c[n] = 1
                c[n-1] = 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            if i == N1-1 or j == N2-1:
                break
            n += 1
            # add only one event, but with coincidence 2 and multiplicity 2
            st[n] = spikes1[i]
            c[n] = 2
            mp[n] = 2

    st = st[:n+2]
    c = c[:n+2]
    mp = mp[:n+2]

    st[0] = spikes1[0]
    st[len(st)-1] = spikes1[len(spikes1)-1]
    c[0] = c[1]
    c[len(c)-1] = c[len(c)-2]
    mp[0] = mp[1]
    mp[len(mp)-1] = mp[len(mp)-2]

    return st, c, mp
