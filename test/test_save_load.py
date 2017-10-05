""" test_save_load.py

Tests saving and loading of spike trains

Copyright 2016, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
from numpy.testing import assert_array_equal

import tempfile
import os.path

import pyspike as spk


def test_save_load():
    file_name = os.path.join(tempfile.mkdtemp(prefix='pyspike_'),
                             "save_load.txt")

    N = 10
    # generate some spike trains
    spike_trains = []
    for n in range(N):
        spike_trains.append(spk.generate_poisson_spikes(1.0, [0, 100]))

    # save them into txt file
    spk.save_spike_trains_to_txt(spike_trains, file_name, precision=17)

    # load again
    spike_trains_loaded = spk.load_spike_trains_from_txt(file_name, [0, 100])

    for n in range(N):
        assert_array_equal(spike_trains[n].spikes,
                           spike_trains_loaded[n].spikes)


if __name__ == "__main__":
    test_save_load()
