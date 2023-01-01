#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_add.pyx

cython implementation of the add function for piece-wise const and 
piece-wise linear functions

Note: using cython memoryviews (e.g. double[:]) instead of ndarray objects
improves the performance of spike_distance by a factor of 10!

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_add.pyx

which gives::

  cython_add.html

"""

import numpy as np
cimport numpy as np

from libc.math cimport fabs

#DTYPE = float
#ctypedef np.float_t DTYPE_t

############################################################
# add_piece_wise_const_cython
############################################################
def add_piece_wise_const_cython(double[:] x1, double[:] y1, 
                                double[:] x2, double[:] y2):

    cdef int N1 = len(x1)
    cdef int N2 = len(x2)
    cdef double[:] x_new = np.empty(N1+N2)
    cdef double[:] y_new = np.empty(N1+N2-1)
    cdef int index1 = 0
    cdef int index2 = 0
    cdef int index = 0
    cdef int i
    with nogil: # release the interpreter lock to allow multi-threading
        x_new[0] = x1[0]
        y_new[0] = y1[0] + y2[0]
        while (index1+1 < N1-1) and (index2+1 < N2-1):
            index += 1
            # print(index1+1, x1[index1+1], y1[index1+1], x_new[index])
            if x1[index1+1] < x2[index2+1]:
                index1 += 1
                x_new[index] = x1[index1]
            elif x1[index1+1] > x2[index2+1]:
                index2 += 1
                x_new[index] = x2[index2]
            else: # x1[index1+1] == x2[index2+1]:
                index1 += 1
                index2 += 1
                x_new[index] = x1[index1]
            y_new[index] = y1[index1] + y2[index2]
        # one array reached the end -> copy the contents of the other to the end
        if index1+1 < N1-1:
            x_new[index+1:index+1+N1-index1-1] = x1[index1+1:]
            for i in xrange(N1-index1-2):
                y_new[index+1+i] = y1[index1+1+i] + y2[N2-2]
            index += N1-index1-2
        elif index2+1 < N2-1:
            x_new[index+1:index+1+N2-index2-1] = x2[index2+1:]
            for i in xrange(N2-index2-2):
                y_new[index+1+i] = y2[index2+1+i] + y1[N1-2]
            index += N2-index2-2
        else: # both arrays reached the end simultaneously
            # only the last x-value missing
            x_new[index+1] = x1[N1-1]
    # end nogil
    # return np.asarray(x_new[:index+2]), np.asarray(y_new[:index+1])
    return np.asarray(x_new[:index+2]), np.asarray(y_new[:index+1])


############################################################
# add_piece_wise_lin_cython
############################################################
def add_piece_wise_lin_cython(double[:] x1, double[:] y11, double[:] y12, 
                              double[:] x2, double[:] y21, double[:] y22):
    cdef int N1 = len(x1)
    cdef int N2 = len(x2)
    cdef double[:] x_new = np.empty(N1+N2)
    cdef double[:] y1_new = np.empty(N1+N2-1)
    cdef double[:] y2_new = np.empty_like(y1_new)
    cdef int index1 = 0 # index for self
    cdef int index2 = 0 # index for f
    cdef int index = 0  # index for new
    cdef int i
    cdef double y
    with nogil: # release the interpreter lock to allow multi-threading
        x_new[0] = x1[0]
        y1_new[0] = y11[0] + y21[0]
        while (index1+1 < N1-1) and (index2+1 < N2-1):
            # print(index1+1, x1[index1+1], self.y[index1+1], x_new[index])
            if x1[index1+1] < x2[index2+1]:
                # first compute the end value of the previous interval
                # linear interpolation of the interval
                y = y21[index2] + (y22[index2]-y21[index2]) * \
                    (x1[index1+1]-x2[index2]) / (x2[index2+1]-x2[index2])
                y2_new[index] = y12[index1] + y
                index1 += 1
                index += 1
                x_new[index] = x1[index1]
                # and the starting value for the next interval
                y1_new[index] = y11[index1] + y
            elif x1[index1+1] > x2[index2+1]:
                # first compute the end value of the previous interval
                # linear interpolation of the interval
                y = y11[index1] + (y12[index1]-y11[index1]) * \
                    (x2[index2+1]-x1[index1]) / \
                    (x1[index1+1]-x1[index1])
                y2_new[index] = y22[index2] + y
                index2 += 1
                index += 1
                x_new[index] = x2[index2]
                # and the starting value for the next interval
                y1_new[index] = y21[index2] + y
            else: # x1[index1+1] == x2[index2+1]:
                y2_new[index] = y12[index1] + y22[index2]
                index1 += 1
                index2 += 1
                index += 1
                x_new[index] = x1[index1]
                y1_new[index] = y11[index1] + y21[index2]
        # one array reached the end -> copy the contents of the other to the end
        if index1+1 < N1-1:
            x_new[index+1:index+1+N1-index1-1] = x1[index1+1:]
            for i in xrange(N1-index1-2):
                # compute the linear interpolations value
                y = y21[index2] + (y22[index2]-y21[index2]) * \
                    (x1[index1+1+i]-x2[index2]) / (x2[index2+1]-x2[index2])
                y1_new[index+1+i] = y11[index1+1+i] + y
                y2_new[index+i] = y12[index1+i] + y
            index += N1-index1-2
        elif index2+1 < N2-1:
            x_new[index+1:index+1+N2-index2-1] = x2[index2+1:]
            # compute the linear interpolations values
            for i in xrange(N2-index2-2):
                y = y11[index1] + (y12[index1]-y11[index1]) * \
                    (x2[index2+1+i]-x1[index1]) / \
                    (x1[index1+1]-x1[index1])
                y1_new[index+1+i] = y21[index2+1+i] + y
                y2_new[index+i] = y22[index2+i] + y
            index += N2-index2-2
        else: # both arrays reached the end simultaneously
            # only the last x-value missing
            x_new[index+1] = x1[N1-1]
        # finally, the end value for the last interval
        y2_new[index] = y12[N1-2]+y22[N2-2]
        # only use the data that was actually filled
    # end nogil
    return (np.asarray(x_new[:index+2]),
            np.asarray(y1_new[:index+1]), 
            np.asarray(y2_new[:index+1]))


############################################################
# add_discrete_function_cython
############################################################
def add_discrete_function_cython(double[:] x1, double[:] y1, double[:] mp1,
                                 double[:] x2, double[:] y2, double[:] mp2):

    cdef double[:] x_new = np.empty(len(x1) + len(x2))
    cdef double[:] y_new = np.empty_like(x_new)
    cdef double[:] mp_new = np.empty_like(x_new)
    cdef int index1 = 0
    cdef int index2 = 0
    cdef int index = 0
    cdef int N1 = len(y1)-1
    cdef int N2 = len(y2)-1
    x_new[0] = x1[0]
    while (index1+1 < N1) and (index2+1 < N2):
        if x1[index1+1] < x2[index2+1]:
            index1 += 1
            index += 1
            x_new[index] = x1[index1]
            y_new[index] = y1[index1]
            mp_new[index] = mp1[index1]
        elif x1[index1+1] > x2[index2+1]:
            index2 += 1
            index += 1
            x_new[index] = x2[index2]
            y_new[index] = y2[index2]
            mp_new[index] = mp2[index2]
        else:  # x1[index1+1] == x2[index2+1]
            index1 += 1
            index2 += 1
            index += 1
            x_new[index] = x1[index1]
            y_new[index] = y1[index1] + y2[index2]
            mp_new[index] = mp1[index1] + mp2[index2]
    # one array reached the end -> copy the contents of the other to the end
    if index1+1 < N1:
        x_new[index+1:index+1+N1-index1] = x1[index1+1:]
        y_new[index+1:index+1+N1-index1] = y1[index1+1:]
        mp_new[index+1:index+1+N1-index1] = mp1[index1+1:]
        index += N1-index1
    elif index2+1 < N2:
        x_new[index+1:index+1+N2-index2] = x2[index2+1:]
        y_new[index+1:index+1+N2-index2] = y2[index2+1:]
        mp_new[index+1:index+1+N2-index2] = mp2[index2+1:]
        index += N2-index2
    else:  # both arrays reached the end simultaneously
        x_new[index+1] = x1[index1+1]
        y_new[index+1] = y1[index1+1] + y2[index2+1]
        mp_new[index+1] = mp1[index1+1] + mp2[index2+1]
        index += 1

    y_new[0] = y_new[1]
    mp_new[0] = mp_new[1]

    # the last value is again the end of the interval
    # only use the data that was actually filled
    return (np.asarray(x_new[:index+1]), 
            np.asarray(y_new[:index+1]), 
            np.asarray(mp_new[:index+1]))
