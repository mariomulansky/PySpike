""" test_regression_11.py

Regression test for Issue 11
https://github.com/mariomulansky/PySpike/issues/11

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

# run as
# python -Qnew test_regression_11.py
# to check correct behavior with new division

# import division to see if everythin works with new division operator
from __future__ import division

import pyspike as spk

M = 19  # uneven number of spike trains

spike_trains = [spk.generate_poisson_spikes(1.0, [0, 100]) for i in xrange(M)]

isi_prof = spk.isi_profile_multi(spike_trains)
isi = spk.isi_distance_multi(spike_trains)

spike_prof = spk.spike_profile_multi(spike_trains)
spike = spk.spike_distance_multi(spike_trains)
