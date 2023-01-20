""" test_sync_filter.py

Tests the spike sync based filtering

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, \
    assert_array_almost_equal

import pyspike as spk
from pyspike import SpikeTrain


def test_single_prof():
    st1 = np.array([1.0, 2.0, 3.0, 4.0])
    st2 = np.array([1.1, 2.1, 3.8])
    st3 = np.array([0.9, 3.1, 4.1])

    # cython implementation
    try:
        from pyspike.cython.cython_profiles import \
            coincidence_single_profile_cython as coincidence_impl
    except ImportError:
        from pyspike.cython.python_backend import \
            coincidence_single_python as coincidence_impl

    sync_prof = spk.spike_sync_profile(SpikeTrain(st1, 5.0),
                                       SpikeTrain(st2, 5.0))

    coincidences = np.array(coincidence_impl(st1, st2, 0, 5.0, 0.0))
    print(coincidences)
    for i, t in enumerate(st1):
        assert_allclose(coincidences[i], sync_prof.y[sync_prof.x == t],
                     err_msg="At index %d" % i)

    coincidences = np.array(coincidence_impl(st2, st1, 0, 5.0, 0.0))
    for i, t in enumerate(st2):
        assert_allclose(coincidences[i], sync_prof.y[sync_prof.x == t],
                     err_msg="At index %d" % i)

    sync_prof = spk.spike_sync_profile(SpikeTrain(st1, 5.0),
                                       SpikeTrain(st3, 5.0))

    coincidences = np.array(coincidence_impl(st1, st3, 0, 5.0, 0.0))
    for i, t in enumerate(st1):
        assert_allclose(coincidences[i], sync_prof.y[sync_prof.x == t],
                     err_msg="At index %d" % i)

    st1 = np.array([1.0, 2.0, 3.0, 4.0])
    st2 = np.array([1.0, 2.0, 4.0])

    sync_prof = spk.spike_sync_profile(SpikeTrain(st1, 5.0),
                                       SpikeTrain(st2, 5.0))

    coincidences = np.array(coincidence_impl(st1, st2, 0, 5.0, 0.0))
    for i, t in enumerate(st1):
        expected = sync_prof.y[sync_prof.x == t]/sync_prof.mp[sync_prof.x == t]
        assert_allclose(coincidences[i], expected,
                     err_msg="At index %d" % i)


def test_filter():
    st1 = SpikeTrain(np.array([1.0, 2.0, 3.0, 4.0]), 5.0)
    st2 = SpikeTrain(np.array([1.1, 2.1, 3.8]), 5.0)
    st3 = SpikeTrain(np.array([0.9, 3.1, 4.1]), 5.0)

    # filtered_spike_trains = spk.filter_by_spike_sync([st1, st2], 0.5)

    # assert_allclose(filtered_spike_trains[0].spikes, [1.0, 2.0, 4.0])
    # assert_allclose(filtered_spike_trains[1].spikes, [1.1, 2.1, 3.8])

    # filtered_spike_trains = spk.filter_by_spike_sync([st2, st1], 0.5)

    # assert_allclose(filtered_spike_trains[0].spikes, [1.1, 2.1, 3.8])
    # assert_allclose(filtered_spike_trains[1].spikes, [1.0, 2.0, 4.0])

    filtered_spike_trains = spk.filter_by_spike_sync([st1, st2, st3], 0.75)

    for st in filtered_spike_trains:
        print(st.spikes)

    assert_allclose(filtered_spike_trains[0].spikes, [1.0, 4.0])
    assert_allclose(filtered_spike_trains[1].spikes, [1.1, 3.8])
    assert_allclose(filtered_spike_trains[2].spikes, [0.9, 4.1])


if __name__ == "__main__":
    test_single_prof()
    test_filter()
