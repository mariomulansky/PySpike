""" averages.py

Simple example showing how to compute averages of distance profiles

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import print_function

import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                              edges=(0, 4000))

f = spk.isi_profile(spike_trains[0], spike_trains[1])

print("ISI-distance: %.8f" % f.avrg())

isi1 = f.avrg(interval=(0, 1000))
isi2 = f.avrg(interval=(1000, 2000))
isi3 = f.avrg(interval=[(0, 1000), (2000, 3000)])
isi4 = f.avrg(interval=[(1000, 2000), (3000, 4000)])

print("ISI-distance (0-1000):                    %.8f" % isi1)
print("ISI-distance (1000-2000):                 %.8f" % isi2)
print("ISI-distance (0-1000) and (2000-3000):    %.8f" % isi3)
print("ISI-distance (1000-2000) and (3000-4000): %.8f" % isi4)
print()

f = spk.spike_profile(spike_trains[0], spike_trains[1])

print("SPIKE-distance: %.8f" % f.avrg())

spike1 = f.avrg(interval=(0, 1000))
spike2 = f.avrg(interval=(1000, 2000))
spike3 = f.avrg(interval=[(0, 1000), (2000, 3000)])
spike4 = f.avrg(interval=[(1000, 2000), (3000, 4000)])

print("SPIKE-distance (0-1000):                    %.8f" % spike1)
print("SPIKE-distance (1000-2000):                 %.8f" % spike2)
print("SPIKE-distance (0-1000) and (2000-3000):    %.8f" % spike3)
print("SPIKE-distance (1000-2000) and (3000-4000): %.8f" % spike4)
print()

f = spk.spike_sync_profile(spike_trains[0], spike_trains[1])

print("SPIKE-Synchronization: %.8f" % f.avrg())

spike_sync1 = f.avrg(interval=(0, 1000))
spike_sync2 = f.avrg(interval=(1000, 2000))
spike_sync3 = f.avrg(interval=[(0, 1000), (2000, 3000)])
spike_sync4 = f.avrg(interval=[(1000, 2000), (3000, 4000)])

print("SPIKE-Sync (0-1000):                        %.8f" % spike_sync1)
print("SPIKE-Sync (1000-2000):                     %.8f" % spike_sync2)
print("SPIKE-Sync (0-1000) and (2000-3000):        %.8f" % spike_sync3)
print("SPIKE-Sync (1000-2000) and (3000-4000):     %.8f" % spike_sync4)
