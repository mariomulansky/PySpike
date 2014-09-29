#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
Doc

To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_distance.pyx

which gives::

  cython_distance.html


"""

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def isi_distance_cython(np.ndarray[DTYPE_t, ndim=1] s1, np.ndarray[DTYPE_t, ndim=1] s2):

    cdef np.ndarray[DTYPE_t, ndim=1] spike_events
    # the values have one entry less - the number of intervals between events
    cdef np.ndarray[DTYPE_t, ndim=1] isi_values
    cdef int index1, index2, index
    cdef int N1, N2
    cdef double nu1, nu2
    N1 = len(s1)-1
    N2 = len(s2)-1

    nu1 = s1[1]-s1[0]
    nu2 = s2[1]-s2[0]
    spike_events = np.empty(N1+N2)
    spike_events[0] = s1[0]
    isi_values = np.empty(N1+N2-1)
    isi_values[0] = (nu1-nu2)/max(nu1,nu2)
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
        isi_values[index] = (nu1 - nu2) / max(nu1, nu2)
        index += 1
    # the last event is the interval end
    spike_events[index] = s1[N1]

    return spike_events[:index+1], isi_values[:index]
