from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("../test/SPIKE_Sync_Test.txt",
                                              time_interval=(0, 4000))

plt.figure()

f = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
x, y = f.get_plottable_data()
plt.plot(x, y, '--ok', label="SPIKE-SYNC profile")
print(f.x)
print(f.y)
print(f.mp)

print("Average:", f.avrg())


f = spk.spike_profile(spike_trains[0], spike_trains[1])
x, y = f.get_plottable_data()

plt.plot(x, y, '-b', label="SPIKE-profile")

plt.axis([0, 4000, -0.1, 1.1])
plt.legend(loc="center right")

plt.figure()

f = spk.spike_sync_profile_multi(spike_trains)
x, y = f.get_plottable_data()
plt.plot(x, y, '-b', alpha=0.7, label="SPIKE-Sync profile")

x1, y1 = f.get_plottable_data(averaging_window_size=50)
plt.plot(x1, y1, '-k', lw=2.5, label="averaged SPIKE-Sync profile")


print("Average:", f.avrg())

plt.show()
