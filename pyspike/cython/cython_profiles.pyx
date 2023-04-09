#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_profiles.pyx

cython implementation of the isi-, spike- and spike-sync profiles

Note: using cython memoryviews (e.g. double[:]) instead of ndarray objects
improves the performance of spike_distance by a factor of 10!

Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_profiles.pyx

which gives::

  cython_profiles.html

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
# isi_profile_cython
############################################################
def isi_profile_cython(double[:] s1, double[:] s2,
                       double t_start, double t_end,
                       double MRTS=0.):

    cdef double[:] spike_events
    cdef double[:] isi_values
    cdef int index1, index2, index
    cdef int N1, N2
    cdef double nu1, nu2
    N1 = len(s1)
    N2 = len(s2)

    spike_events = np.empty(N1+N2+2)
    # the values have one entry less as they are defined at the intervals
    isi_values = np.empty(N1+N2+1)

    # first x-value of the profile
    spike_events[0] = t_start

    # first interspike interval - check if a spike exists at the start time
    if s1[0] > t_start:
        # edge correction
        nu1 = fmax(s1[0]-t_start, s1[1]-s1[0]) if N1 > 1 else s1[0]-t_start
        index1 = -1
    else:
        nu1 = s1[1]-s1[0] if N1 > 1 else t_end-s1[0]
        index1 = 0

    if s2[0] > t_start:
        # edge correction
        nu2 = fmax(s2[0]-t_start, s2[1]-s2[0]) if N2 > 1 else s2[0]-t_start
        index2 = -1
    else:
        nu2 = s2[1]-s2[0] if N2 > 1 else t_end-s2[0]
        index2 = 0

    isi_values[0] = fabs(nu1-nu2)/fmax(MRTS, fmax(nu1, nu2))
    index = 1

    with nogil: # release the interpreter to allow multithreading
        while index1+index2 < N1+N2-2:
            # check which spike is next, only if there are spikes left in 1
            # next spike in 1 is earlier, or there are no spikes left in 2
            if (index1 < N1-1) and ((index2 == N2-1) or
                                    (s1[index1+1] < s2[index2+1])):
                index1 += 1
                spike_events[index] = s1[index1]
                if index1 < N1-1:
                    nu1 = s1[index1+1]-s1[index1]
                else:
                    # edge correction
                    nu1 = fmax(t_end-s1[index1], nu1) if N1 > 1 \
                          else t_end-s1[index1]
            elif (index2 < N2-1) and ((index1 == N1-1) or
                                      (s1[index1+1] > s2[index2+1])):
                index2 += 1
                spike_events[index] = s2[index2]
                if index2 < N2-1:
                    nu2 = s2[index2+1]-s2[index2]
                else:
                    # edge correction
                    nu2 = fmax(t_end-s2[index2], nu2) if N2 > 1 \
                          else t_end-s2[index2]
            else: # s1[index1+1] == s2[index2+1]
                index1 += 1
                index2 += 1
                spike_events[index] = s1[index1]
                if index1 < N1-1:
                    nu1 = s1[index1+1]-s1[index1]
                else:
                    # edge correction
                    nu1 = fmax(t_end-s1[index1], nu1) if N1 > 1 \
                          else t_end-s1[index1]
                if index2 < N2-1:
                    nu2 = s2[index2+1]-s2[index2]
                else:
                    # edge correction
                    nu2 = fmax(t_end-s2[index2], nu2) if N2 > 1 \
                          else t_end-s2[index2]
            # compute the corresponding isi-distance
            isi_values[index] = fabs(nu1 - nu2) / fmax(MRTS, fmax(nu1, nu2))
            index += 1
        # the last event is the interval end
        if spike_events[index-1] == t_end:
            index -= 1
        else:
            spike_events[index] = t_end
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
                                       int start_index,
                                       double t_start, double t_end) nogil:
    """ Returns the minimal distance |spike_time - spike_train[i]| 
    with i>=start_index.
    """
    cdef double d, d_temp
    # start with the distance to the start time
    d = fabs(spike_time - t_start)
    if start_index < 0:
        start_index = 0
    while start_index < N:
        d_temp = fabs(spike_time - spike_train[start_index])
        if d_temp > d:
            return d
        else:
            d = d_temp
        start_index += 1

    # finally, check the distance to end time
    d_temp = fabs(t_end - spike_time)
    if d_temp > d:
        return d
    else:
        return d_temp


############################################################
# dist_at_t
############################################################
cdef inline double dist_at_t(double isi1, double isi2, 
                              double s1, double s2,
                              double MRTS, int RI) nogil:
    """ Compute instantaneous Spike Distance
            In: isi1, isi2 - spike time differences around current times in each trains
                s1, s2 - weighted spike time differences between trains
                MRTS -minimum relevant time scal (0 for legacy logic)
                RI - Rate Independent Adaptive spike distance 
                        (False for legacy SPIKE distance)
            Out: Spike Distance at current time
    """
    cdef double meanISI = .5*(isi1+isi2)
    cdef double limitedISI = max(MRTS, meanISI)
    if RI:
        return .5*(s1+s2)/limitedISI
    else:
        return .5*(s1*isi2 + s2*isi1)/(meanISI*limitedISI)

############################################################
# spike_profile_cython
############################################################
def spike_profile_cython(double[:] t1, double[:] t2,
                         double t_start, double t_end,
                         double MRTS=0., int RI=0):

    cdef double[:] spike_events
    cdef double[:] y_starts
    cdef double[:] y_ends
    cdef double[:] t_aux1 = np.empty(2)
    cdef double[:] t_aux2 = np.empty(2)

    cdef int N1, N2, index1, index2, index
    cdef double t_p1, t_f1, t_p2, t_f2, dt_p1, dt_p2, dt_f1, dt_f2
    cdef double isi1, isi2, s1, s2

    N1 = len(t1)
    N2 = len(t2)

    # we can assume at least one spikes per spike train
    assert N1 > 0
    assert N2 > 0

    spike_events = np.empty(N1+N2+2)

    y_starts = np.empty(len(spike_events)-1)
    y_ends = np.empty(len(spike_events)-1)

    with nogil: # release the interpreter to allow multithreading
        spike_events[0] = t_start
        # t_p1 = t_start
        # t_p2 = t_start
        # auxiliary spikes for edge correction - consistent with first/last ISI 
        t_aux1[0] = fmin(t_start, 2*t1[0]-t1[1]) if N1 > 1 else t_start
        t_aux1[1] = fmax(t_end, 2*t1[N1-1]-t1[N1-2]) if N1 > 1 else t_end
        t_aux2[0] = fmin(t_start, 2*t2[0]-t2[1]) if N2 > 1 else t_start
        t_aux2[1] = fmax(t_end, 2*t2[N2-1]-t2[N2-2]) if N2 > 1 else t_end
        t_p1 = t_start if (t1[0] == t_start) else t_aux1[0]
        t_p2 = t_start if (t2[0] == t_start) else t_aux2[0]
        if t1[0] > t_start:
            # dt_p1 = t2[0]-t_start
            t_f1 = t1[0]
            dt_f1 = get_min_dist_cython(t_f1, t2, N2, 0, t_aux2[0], t_aux2[1])
            isi1 = fmax(t_f1-t_start, t1[1]-t1[0]) if N1 > 1 else t_f1-t_start
            dt_p1 = dt_f1
            # s1 = dt_p1*(t_f1-t_start)/isi1
            s1 = dt_p1
            index1 = -1
        else:
            t_f1 = t1[1] if N1 > 1 else t_end
            dt_f1 = get_min_dist_cython(t_f1, t2, N2, 0, t_aux2[0], t_aux2[1])
            dt_p1 = get_min_dist_cython(t_p1, t2, N2, 0, t_aux2[0], t_aux2[1])
            isi1 = t_f1-t1[0]
            s1 = dt_p1
            index1 = 0
        if t2[0] > t_start:
            # dt_p1 = t2[0]-t_start
            t_f2 = t2[0]
            dt_f2 = get_min_dist_cython(t_f2, t1, N1, 0, t_aux1[0], t_aux1[1])
            dt_p2 = dt_f2
            isi2 = fmax(t_f2-t_start, t2[1]-t2[0]) if N2 > 1 else t_f2-t_start
            # s2 = dt_p2*(t_f2-t_start)/isi2
            s2 = dt_p2
            index2 = -1
        else:
            t_f2 = t2[1] if N2 > 1 else t_end
            dt_f2 = get_min_dist_cython(t_f2, t1, N1, 0, t_aux1[0], t_aux1[1])
            dt_p2 = get_min_dist_cython(t_p2, t1, N1, 0, t_aux1[0], t_aux1[1])
            isi2 = t_f2-t2[0]
            s2 = dt_p2
            index2 = 0

        y_starts[0] = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
        index = 1

        while index1+index2 < N1+N2-2:
            # print(index, index1, index2)
            if (index1 < N1-1) and (t_f1 < t_f2 or index2 == N2-1):
                index1 += 1
                # first calculate the previous interval end value
                s1 = dt_f1*(t_f1-t_p1) / isi1
                # the previous time now was the following time before:
                dt_p1 = dt_f1
                t_p1 = t_f1    # t_p1 contains the current time point
                # get the next time
                if index1 < N1-1:
                    t_f1 = t1[index1+1]
                else:
                    t_f1 = t_aux1[1]
                spike_events[index] = t_p1
                s2 = (dt_p2*(t_f2-t_p1) + dt_f2*(t_p1-t_p2)) / isi2
                y_ends[index-1] = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
                # now the next interval start value
                if index1 < N1-1:
                    dt_f1 = get_min_dist_cython(t_f1, t2, N2, index2,
                                                t_aux2[0], t_aux2[1])
                    isi1 = t_f1-t_p1
                    s1 = dt_p1
                else:
                    dt_f1 = dt_p1
                    isi1 = fmax(t_end-t1[N1-1], t1[N1-1]-t1[N1-2]) if N1 > 1 \
                           else t_end-t1[N1-1]
                    # s1 needs adjustment due to change of isi1
                    # s1 = dt_p1*(t_end-t1[N1-1])/isi1
                    # Eero's correction: no adjustment
                    s1 = dt_p1
                # s2 is the same as above, thus we can compute y2 immediately
                y_starts[index] = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
            elif (index2 < N2-1) and (t_f1 > t_f2 or index1 == N1-1):
                index2 += 1
                # first calculate the previous interval end value
                s2 = dt_f2*(t_f2-t_p2) / isi2
                # the previous time now was the following time before:
                dt_p2 = dt_f2
                t_p2 = t_f2    # t_p2 contains the current time point
                # get the next time
                if index2 < N2-1:
                    t_f2 = t2[index2+1]
                else:
                    t_f2 = t_aux2[1]
                spike_events[index] = t_p2
                s1 = (dt_p1*(t_f1-t_p2) + dt_f1*(t_p2-t_p1)) / isi1
                y_ends[index-1] = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
                # now the next interval start value
                if index2 < N2-1:
                    dt_f2 = get_min_dist_cython(t_f2, t1, N1, index1,
                                                t_aux1[0], t_aux1[1])
                    isi2 = t_f2-t_p2
                    s2 = dt_p2
                else:
                    dt_f2 = dt_p2
                    isi2 = fmax(t_end-t2[N2-1], t2[N2-1]-t2[N2-2]) if N2 > 1 \
                           else t_end-t2[N2-1]
                    # s2 needs adjustment due to change of isi2
                    # s2 = dt_p2*(t_end-t2[N2-1])/isi2
                    # Eero's correction: no adjustment
                    s2 = dt_p2
                # s2 is the same as above, thus we can compute y2 immediately
                y_starts[index] = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
            else: # t_f1 == t_f2 - generate only one event
                index1 += 1
                index2 += 1
                t_p1 = t_f1
                t_p2 = t_f2
                dt_p1 = 0.0
                dt_p2 = 0.0
                spike_events[index] = t_f1
                y_ends[index-1] = 0.0
                y_starts[index] = 0.0
                if index1 < N1-1:
                    t_f1 = t1[index1+1]
                    dt_f1 = get_min_dist_cython(t_f1, t2, N2, index2,
                                                t_aux2[0], t_aux2[1])
                    isi1 = t_f1 - t_p1
                else:
                    t_f1 = t_aux1[1]
                    dt_f1 = dt_p1
                    isi1 = fmax(t_end-t1[N1-1], t1[N1-1]-t1[N1-2]) if N1 > 1 \
                           else t_end-t1[N1-1]
                if index2 < N2-1:
                    t_f2 = t2[index2+1]
                    dt_f2 = get_min_dist_cython(t_f2, t1, N1, index1,
                                                t_aux1[0], t_aux1[1])
                    isi2 = t_f2 - t_p2
                else:
                    t_f2 = t_aux2[1]
                    dt_f2 = dt_p2
                    isi2 = fmax(t_end-t2[N2-1], t2[N2-1]-t2[N2-2]) if N2 > 1 \
                           else t_end-t2[N2-1]
            index += 1
        # the last event is the interval end
        if spike_events[index-1] == t_end:
            index -= 1
        else:
            spike_events[index] = t_end
            s1 = dt_f1
            s2 = dt_f2
            y_ends[index-1] = dist_at_t(isi1, isi2, s1, s2, MRTS, RI)
    # end nogil

    # use only the data added above 
    # could be less than original length due to equal spike times
    return spike_events[:index+1], y_starts[:index], y_ends[:index]



############################################################
# coincidence_profile_cython
############################################################
def coincidence_profile_cython(double[:] spikes1, double[:] spikes2,
                               double t_start, double t_end, double max_tau, double MRTS=0):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int i = -1
    cdef int j = -1
    cdef int n = 0
    cdef double[:] st = np.zeros(N1 + N2 + 2)  # spike times
    cdef double[:] c = np.zeros(N1 + N2 + 2)   # coincidences
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
                # both get marked with 1
                c[n] = 1
                c[n-1] = 1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            st[n] = spikes2[j]
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # both get marked with 1
                c[n] = 1
                c[n-1] = 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            n += 1
            # add only one event, but with coincidence 2 and multiplicity 2
            st[n] = spikes1[i]
            c[n] = 2
            mp[n] = 2

    st = st[:n+2]
    c = c[:n+2]
    mp = mp[:n+2]

    st[0] = t_start
    st[len(st)-1] = t_end
    if N1 + N2 > 0:
        c[0] = c[1]
        c[len(c)-1] = c[len(c)-2]
        mp[0] = mp[1]
        mp[len(mp)-1] = mp[len(mp)-2]
    else:
        c[0] = 1
        c[1] = 1

    return st, c, mp


############################################################
# coincidence_single_profile_cython
############################################################
def coincidence_single_profile_cython(double[:] spikes1, double[:] spikes2,
                                      double t_start, double t_end, double max_tau, double MRTS=0.):

    cdef int N1 = len(spikes1)
    cdef int N2 = len(spikes2)
    cdef int j = -1
    cdef double[:] c = np.zeros(N1)   # coincidences
    cdef double interval = t_end - t_start
    cdef double tau

    cdef double true_max = t_end - t_start
    if max_tau > 0:
        true_max = fmin(true_max, 2*max_tau)

    for i in xrange(N1):
        while j < N2-1 and spikes2[j+1] < spikes1[i]:
            # move forward until spikes2[j] is the last spike before spikes1[i]
            # note that if spikes2[j] is after spikes1[i] we dont do anything
            j += 1
        tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
        if j > -1 and fabs(spikes1[i]-spikes2[j]) < tau:
            # current spike in st1 is coincident
            c[i] = 1
        if j < N2-1 and (j < 0 or spikes2[j] < spikes1[i]):
            # in case spikes2[j] is before spikes1[i] it has to be the one 
            # right before (see above), hence we move one forward and also 
            # check the next spike
            j += 1
            tau = get_tau(spikes1, spikes2, i, j, true_max, MRTS)
            if fabs(spikes2[j]-spikes1[i]) < tau:
                # current spike in st1 is coincident
                c[i] = 1
    return c
