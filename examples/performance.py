"""
Compute distances of large sets of spike trains for performance tests

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function

import pyspike as spk
from datetime import datetime
import cProfile
import pstats

M = 100    # number of spike trains
r = 1.0    # rate of Poisson spike times
T = 1E3    # length of spike trains

print("%d spike trains with %d spikes" % (M, int(r*T)))

spike_trains = []

t_start = datetime.now()
for i in range(M):
    spike_trains.append(spk.generate_poisson_spikes(r, T))
t_end = datetime.now()
runtime = (t_end-t_start).total_seconds()

print("Spike generation runtime: %.3fs" % runtime)
print()

print("================ ISI COMPUTATIONS ================")
print("    MULTIVARIATE DISTANCE")
cProfile.run('spk.isi_distance_multi(spike_trains)', 'performance.stat')
p = pstats.Stats('performance.stat')
p.strip_dirs().sort_stats('tottime').print_stats(5)

print("    MULTIVARIATE PROFILE")
cProfile.run('spk.isi_profile_multi(spike_trains)', 'performance.stat')
p = pstats.Stats('performance.stat')
p.strip_dirs().sort_stats('tottime').print_stats(5)

print("================ SPIKE COMPUTATIONS ================")
print("    MULTIVARIATE DISTANCE")
cProfile.run('spk.spike_distance_multi(spike_trains)', 'performance.stat')
p = pstats.Stats('performance.stat')
p.strip_dirs().sort_stats('tottime').print_stats(5)

print("    MULTIVARIATE PROFILE")
cProfile.run('spk.spike_profile_multi(spike_trains)', 'performance.stat')
p = pstats.Stats('performance.stat')
p.strip_dirs().sort_stats('tottime').print_stats(5)

print("================ SPIKE-SYNC COMPUTATIONS ================")
print("    MULTIVARIATE DISTANCE")
cProfile.run('spk.spike_sync_multi(spike_trains)', 'performance.stat')
p = pstats.Stats('performance.stat')
p.strip_dirs().sort_stats('tottime').print_stats(5)

print("    MULTIVARIATE PROFILE")
cProfile.run('spk.spike_sync_profile_multi(spike_trains)', 'performance.stat')
p = pstats.Stats('performance.stat')
p.strip_dirs().sort_stats('tottime').print_stats(5)
