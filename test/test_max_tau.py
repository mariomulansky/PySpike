""" test_max_tau.py

Tests that max_tau bounds the coincidence window for pairs of spikes that are
interior to their own trains, not only for spikes at the trains' edges.

Copyright 2026, Tony DeFazio

Distributed under the BSD License

"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, \
    assert_array_less

import pyspike as spk
from pyspike import SpikeTrain


def gen_spike_trains():
    """Two trains whose six pairs are separated by 0, 0.1, 0.2, 0.3, 0.4 and
    0.5, so raising max_tau past each separation admits exactly one more pair.
    Every expected value below is a sixth, countable by hand off these two
    lines.
    """
    return (SpikeTrain([0.0, 1.0, 3.0, 5.0, 7.0, 9.0], 10.0),
            SpikeTrain([0.0, 1.1, 3.2, 5.3, 7.4, 9.5], 10.0))


def test_spike_sync_max_tau_bounds_interior_pairs():
    spikes1, spikes2 = gen_spike_trains()

    # Unaided, every pair's window is at least half its own surrounding ISI --
    # 1.0 at the widest pair -- so all six match.
    assert_almost_equal(spk.spike_sync(spikes1, spikes2), 1.0, decimal=15)

    # Each cap sits between two separations, so it admits one further pair.
    # The first row is set by the pair at separation 0, which the equal-times
    # branch admits without calling get_tau at all -- that row is insensitive
    # to the cap rather than a test of it.
    for max_tau, expected in ((0.05, 1.0 / 6),
                              (0.15, 2.0 / 6),
                              (0.25, 3.0 / 6),
                              (0.35, 4.0 / 6),
                              (0.45, 5.0 / 6),
                              (0.55, 6.0 / 6)):
        assert_almost_equal(spk.spike_sync(spikes1, spikes2, max_tau=max_tau),
                            expected, decimal=15)


def test_spike_sync_profile_max_tau_bounds_interior_pairs():
    spikes1, spikes2 = gen_spike_trains()

    # Under a 0.25 cap the pairs at 0, 0.1 and 0.2 coincide and the rest do
    # not. Each of those later pairs is interior to both trains, which is the
    # case a cap has to reach and did not.
    expected_x = np.array([0.0, 0.0, 1.0, 1.1, 3.0, 3.2,
                           5.0, 5.3, 7.0, 7.4, 9.0, 9.5, 10.0])
    expected_y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    f = spk.spike_sync_profile(spikes1, spikes2, max_tau=0.25)

    # Both trains fire at 0.0, so that entry carries multiplicity 2 and the
    # raw sum f.y reads 2.0 there. get_plottable_data returns f.y/f.mp, which
    # is the coincidence indicator this test is about.
    x, y = f.get_plottable_data()
    assert_array_almost_equal(x, expected_x, decimal=15)
    assert_array_almost_equal(y, expected_y, decimal=15)


def test_spike_directionality_max_tau():
    # get_tau serves spike-directionality and spike-train-order too, so the
    # cap has to reach them by the same route.
    spikes1, spikes2 = gen_spike_trains()

    # spike_directionality normalizes by len(spikes1), which is 6 here. Five
    # of the six spikes of train 1 lead their partner; the pair at 0.0 has
    # zero lag and contributes nothing.
    assert_almost_equal(spk.spike_directionality(spikes1, spikes2),
                        5.0 / 6, decimal=15)
    assert_almost_equal(
        spk.spike_directionality(spikes1, spikes2, max_tau=0.25),
        2.0 / 6, decimal=15)
    assert_almost_equal(
        spk.spike_directionality(spikes1, spikes2, max_tau=0.35),
        3.0 / 6, decimal=15)


def test_spike_sync_max_tau_is_strictly_increasing():
    # Each of these caps clears one more pair separation, so every step must
    # move. A merely non-decreasing check would pass on the unbounded
    # behavior: without the cap only the last, edge-of-train pair responds to
    # max_tau, so the sequence rises exactly once (5/6 -> 6/6).
    spikes1, spikes2 = gen_spike_trains()
    values = np.array([spk.spike_sync(spikes1, spikes2, max_tau=t)
                       for t in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55)])
    assert_array_less(0.0, np.diff(values))


def test_no_cap_is_unchanged():
    # The property the patch promises to preserve, and the one a later
    # refactor is most likely to break: 0 and None mean no upper bound, and
    # must return what an uncapped call returns.
    spikes1 = SpikeTrain([0.4, 1.9, 4.1, 6.6, 8.0], 10.0)
    spikes2 = SpikeTrain([0.9, 2.7, 3.3, 7.2, 9.4], 10.0)

    uncapped = spk.spike_sync(spikes1, spikes2)
    assert_almost_equal(spk.spike_sync(spikes1, spikes2, max_tau=0),
                        uncapped, decimal=15)
    assert_almost_equal(spk.spike_sync(spikes1, spikes2, max_tau=None),
                        uncapped, decimal=15)


def test_max_tau_bounds_an_mrts_raised_window():
    # MRTS raises the window off the local ISIs, which is what it is for. The
    # cap has to survive that: without it, a seeded slot enters Interpolate as
    # an argument rather than as an outer bound and the interpolation walks
    # straight past it.
    spikes1 = SpikeTrain([0.0, 0.1, 2.1, 2.2, 4.2, 4.3], 6.0)
    spikes2 = SpikeTrain([0.4, 0.5, 2.5, 2.6, 4.6, 4.7], 6.0)

    # The nearest pair across these trains is 0.3 apart, so a 0.2 cap admits
    # nothing however far MRTS lifts the window.
    assert_almost_equal(
        spk.spike_sync(spikes1, spikes2, max_tau=0.2, MRTS=2.0),
        0.0, decimal=15)

    # A cap wider than MRTS raises must not claw the window back down.
    assert_almost_equal(
        spk.spike_sync(spikes1, spikes2, max_tau=0.5, MRTS=2.0),
        spk.spike_sync(spikes1, spikes2, MRTS=2.0), decimal=15)


if __name__ == "__main__":
    test_spike_sync_max_tau_bounds_interior_pairs()
    test_spike_sync_profile_max_tau_bounds_interior_pairs()
    test_spike_directionality_max_tau()
    test_spike_sync_max_tau_is_strictly_increasing()
    test_no_cap_is_unchanged()
    test_max_tau_bounds_an_mrts_raised_window()
