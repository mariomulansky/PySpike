#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_simulated_annealing.pyx

cython implementation of a simulated annealing algorithm to find the optimal
spike train order

Note: using cython memoryviews (e.g. double[:]) instead of ndarray objects
improves the performance of spike_distance by a factor of 10!

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_simulated_annealing.pyx

which gives:

  cython_simulated_annealing.html

"""

import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport fmod
from libc.stdlib cimport rand
from libc.stdlib cimport RAND_MAX

#DTYPE = float
#ctypedef np.float_t DTYPE_t

def sim_ann_cython(double[:, :] D, double T_start, double T_end, double alpha):

    cdef long N = len(D)
    cdef double A = np.sum(np.triu(D, 0))
    cdef long[:] p = np.arange(N)
    cdef double T = T_start
    cdef long iterations
    cdef long succ_iter
    cdef long total_iter = 0
    cdef double delta_A
    cdef long ind1
    cdef long ind2

    while T > T_end:
        iterations = 0
        succ_iter = 0
        # equilibrate for 100*N steps or 10*N successful steps
        while iterations < 100*N and succ_iter < 10*N:
            # exchange two rows and cols
            # ind1 = np.random.randint(N-1)
            ind1 = rand() % (N-1)
            if ind1 < N-1:
                ind2 = ind1+1
            else:  # this can never happen!
                ind2 = 0
            delta_A = -2*D[p[ind1], p[ind2]]
            if delta_A > 0.0 or exp(delta_A/T) > ((1.0*rand()) / RAND_MAX):
                # swap indices
                p[ind1], p[ind2] = p[ind2], p[ind1]
                A += delta_A
                succ_iter += 1
            iterations += 1
        total_iter += iterations
        T *= alpha   # cool down
        if succ_iter == 0:
            # no successful step -> we believe we have converged
            break

    return p, A, total_iter
