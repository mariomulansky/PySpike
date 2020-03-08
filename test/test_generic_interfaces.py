""" test_generic_interface.py

Tests the generic interfaces of the profile and distance functions

Copyright 2016, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
from numpy.testing import assert_allclose

import pyspike as spk
from pyspike import SpikeTrain


class dist_from_prof:
    """ Simple functor that turns profile function into distance function by
    calling profile.avrg().
    """
    def __init__(self, prof_func):
        self.prof_func = prof_func

    def __call__(self, *args, **kwargs):
        if "interval" in kwargs:
            # forward interval arg into avrg function
            interval = kwargs.pop("interval")
            return self.prof_func(*args, **kwargs).avrg(interval=interval)
        else:
            return self.prof_func(*args, **kwargs).avrg()


def check_func(dist_func):
    """ generic checker that tests the given distance function.
    """
    # generate spike trains:
    t1 = SpikeTrain([0.2, 0.4, 0.6, 0.7], 1.0)
    t2 = SpikeTrain([0.3, 0.45, 0.8, 0.9, 0.95], 1.0)
    t3 = SpikeTrain([0.2, 0.4, 0.6], 1.0)
    t4 = SpikeTrain([0.1, 0.4, 0.5, 0.6], 1.0)
    spike_trains = [t1, t2, t3, t4]

    isi12 = dist_func(t1, t2)
    isi12_ = dist_func([t1, t2])
    assert_allclose(isi12, isi12_)

    isi12_ = dist_func(spike_trains, indices=[0, 1])
    assert_allclose(isi12, isi12_)

    isi123 = dist_func(t1, t2, t3)
    isi123_ = dist_func([t1, t2, t3])
    assert_allclose(isi123, isi123_)

    isi123_ = dist_func(spike_trains, indices=[0, 1, 2])
    assert_allclose(isi123, isi123_)

    # run the same test with an additional interval parameter

    isi12 = dist_func(t1, t2, interval=[0.0, 0.5])
    isi12_ = dist_func([t1, t2], interval=[0.0, 0.5])
    assert_allclose(isi12, isi12_)

    isi12_ = dist_func(spike_trains, indices=[0, 1], interval=[0.0, 0.5])
    assert_allclose(isi12, isi12_)

    isi123 = dist_func(t1, t2, t3, interval=[0.0, 0.5])
    isi123_ = dist_func([t1, t2, t3], interval=[0.0, 0.5])
    assert_allclose(isi123, isi123_)

    isi123_ = dist_func(spike_trains, indices=[0, 1, 2], interval=[0.0, 0.5])
    assert_allclose(isi123, isi123_)


def test_isi_profile():
    check_func(dist_from_prof(spk.isi_profile))


def test_isi_distance():
    check_func(spk.isi_distance)


def test_spike_profile():
    check_func(dist_from_prof(spk.spike_profile))


def test_spike_distance():
    check_func(spk.spike_distance)


def test_spike_sync_profile():
    check_func(dist_from_prof(spk.spike_sync_profile))


def test_spike_sync():
    check_func(spk.spike_sync)


if __name__ == "__main__":
    test_isi_profile()
    test_isi_distance()
    test_spike_profile()
    test_spike_distance()
    test_spike_sync_profile()
    test_spike_sync()
