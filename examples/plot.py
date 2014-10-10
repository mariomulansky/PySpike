""" plot.py

Simple example showing how to load and plot spike trains and their distances.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the MIT License (MIT)
"""


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 
                                              time_interval=(0,4000))

# plot the spike time
for (i,spikes) in enumerate(spike_trains):
    plt.plot(spikes, i*np.ones_like(spikes), 'o')

f = spk.isi_distance(spike_trains[0], spike_trains[1])
x, y = f.get_plottable_data()

plt.figure()
plt.plot(x, np.abs(y), '--k')

print("Average: %.8f" % f.avrg())
print("Absolute average: %.8f" % f.abs_avrg())


f = spk.spike_distance(spike_trains[0], spike_trains[1])
x, y = f.get_plottable_data()
print(x)
print(y)
#plt.figure()
plt.plot(x, y, '-b')

plt.show()
