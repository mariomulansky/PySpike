""" plot.py

Simple example showing how to load and plot spike trains and their distance
profiles.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk


spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                              edges=(0, 4000))

# plot the spike times
for (i, spike_train) in enumerate(spike_trains):
    plt.scatter(spike_train, i*np.ones_like(spike_train), marker='|')

# profile of the first two spike trains
f = spk.isi_profile(spike_trains, indices=[0, 1])
x, y = f.get_plottable_data()

plt.figure()
plt.plot(x, np.abs(y), '--k', label="ISI-profile")

print("ISI-distance: %.8f" % f.avrg())

f = spk.spike_profile(spike_trains, indices=[0, 1])
x, y = f.get_plottable_data()

plt.plot(x, y, '-b', label="SPIKE-profile")

print("SPIKE-distance: %.8f" % f.avrg())

plt.legend(loc="upper left")

plt.show()
