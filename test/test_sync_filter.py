""" test_sync_filter.py

Tests the spike sync based filtering

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, \
    assert_array_almost_equal

import pyspike as spk
from pyspike import SpikeTrain


def test_cython():
    st1 = np.array([1.0, 2.0, 3.0, 4.0])
    st2 = np.array([1.1, 2.1, 3.8])

    # cython implementation
    try:
        from pyspike.cython.cython_profiles import coincidence_single_profile_cython \
            as coincidence_impl
    except ImportError:
        from pyspike.cython.python_backend import coincidence_single_profile_python \
            as coincidence_impl

    sync_prof = spk.spike_sync_profile(SpikeTrain(st1, 5.0),
                                       SpikeTrain(st2, 5.0))

    coincidences = np.array(coincidence_impl(st1, st2, 0, 5.0, 0.0))
    for i, t in enumerate(st1):
        assert_equal(coincidences[i], sync_prof.y[sync_prof.x == t],
                     "At index %d" % i)

    coincidences = np.array(coincidence_impl(st2, st1, 0, 5.0, 0.0))
    for i, t in enumerate(st2):
        assert_equal(coincidences[i], sync_prof.y[sync_prof.x == t],
                     "At index %d" % i)
