""" profiles.py

Simple example showing some functionality of distance profiles.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""


from __future__ import print_function

import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                              edges=(0, 4000))

##### ISI PROFILES #######

# compute the ISI profile of the first two spike trains
f = spk.isi_profile(spike_trains[0], spike_trains[1])

# ISI values at certain points
t = 1200
print("ISI value at t =", t, ":", f(t))
t = [900, 1100, 2000, 3100]
print("ISI value at t =", t, ":", f(t))
print("Average ISI distance:", f.avrg())
print()

# compute the multivariate ISI profile
f = spk.isi_profile(spike_trains)

t = 1200
print("Multivariate ISI value at t =", t, ":", f(t))
t = [900, 1100, 2000, 3100]
print("Multivariate ISI value at t =", t, ":", f(t))
print("Average multivariate ISI distance:", f.avrg())
print()
print()

# for plotting, use the get_plottable_data() member function, see plot.py


##### SPIKE PROFILES #######

# compute the SPIKE profile of the first two spike trains
f = spk.spike_profile(spike_trains[0], spike_trains[1])

# SPIKE distance values at certain points
t = 1200
print("SPIKE value at t =", t, ":", f(t))
t = [900, 1100, 2000, 3100]
print("SPIKE value at t =", t, ":", f(t))
print("Average SPIKE distance:", f.avrg())
print()

# compute the multivariate SPIKE profile
f = spk.spike_profile(spike_trains)

# SPIKE values at certain points
t = 1200
print("Multivariate SPIKE value at t =", t, ":", f(t))
t = [900, 1100, 2000, 3100]
print("Multivariate SPIKE value at t =", t, ":", f(t))
print("Average multivariate SPIKE distance:", f.avrg())
