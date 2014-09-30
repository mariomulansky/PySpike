# performance measure of the isi calculation

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial

import pyspike as spk

def measure_perf(func, loops=10):
    times = np.empty(loops)
    for i in xrange(loops):
        start = time.clock()
        func()
        times[i] = time.clock() - start
    return np.min(times)

print("# approximate number of spikes\tcython time [ms]\tpython time [ms]")

# max times
Ns = np.arange(10000, 50001, 10000)
for N in Ns:

    # first generate some data
    times = 2.0*np.random.random(1.1*N)
    t1 = np.cumsum(times)
    # only up to T
    t1 = spk.add_auxiliary_spikes(t1[t1<N], N)

    times = 2.0*np.random.random(N)
    t2 = np.cumsum(times)
    # only up to T
    t2 = spk.add_auxiliary_spikes(t2[t2<N], N)

    t_cython = measure_perf(partial(spk.spike_distance, t1, t2))

    t_python = measure_perf(partial(spk.distances.spike_distance_python, 
                                    t1, t2))
    
    print("%d\t%.3f\t%.1f" % (N, t_cython*1000, t_python*1000))
