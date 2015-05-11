""" test_empty.py

Tests the distance measure for empty spike trains

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
import numpy as np
from copy import copy
from numpy.testing import assert_equal, assert_almost_equal, \
    assert_array_equal, assert_array_almost_equal

import pyspike as spk
from pyspike import SpikeTrain


def test_get_non_empty():
    st = SpikeTrain([], edges=(0.0, 1.0))
    spikes = st.get_spikes_non_empty()
    assert_array_equal(spikes, [0.0, 1.0])

    st = SpikeTrain([0.5, ], edges=(0.0, 1.0))
    spikes = st.get_spikes_non_empty()
    assert_array_equal(spikes, [0.0, 0.5, 1.0])


def test_isi_empty():
    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([], edges=(0.0, 1.0))
    d = spk.isi_distance(st1, st2)
    assert_equal(d, 0.0)
    prof = spk.isi_profile(st1, st2)
    assert_equal(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 1.0])
    assert_array_equal(prof.y, [0.0, ])

    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.isi_distance(st1, st2)
    assert_equal(d, 0.6*0.4+0.4*0.6)
    prof = spk.isi_profile(st1, st2)
    assert_equal(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 0.4, 1.0])
    assert_array_equal(prof.y, [0.6, 0.4])

    st1 = SpikeTrain([0.6, ], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.isi_distance(st1, st2)
    assert_almost_equal(d, 0.2/0.6*0.4 + 0.0 + 0.2/0.6*0.4, decimal=15)
    prof = spk.isi_profile(st1, st2)
    assert_equal(d, prof.avrg())
    assert_array_almost_equal(prof.x, [0.0, 0.4, 0.6, 1.0], decimal=15)
    assert_array_almost_equal(prof.y, [0.2/0.6, 0.0, 0.2/0.6], decimal=15)


if __name__ == "__main__":
    test_get_non_empty()
    test_isi_empty()
