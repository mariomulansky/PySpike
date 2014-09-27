""" test_function.py

Tests the PieceWiseConst and PieceWiseLinear functions

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

from __future__ import print_function
import numpy as np
from copy import copy
from numpy.testing import assert_equal, assert_almost_equal, \
    assert_array_almost_equal

import pyspike as spk

def test_pwc():
    # some random data
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [1.0, -0.5, 1.5, 0.75]
    f = spk.PieceWiseConstFunc(x, y)
    xp, yp = f.get_plottable_data()
    
    xp_expected = [0.0, 1.0, 1.0, 2.0, 2.0, 2.5, 2.5, 4.0]
    yp_expected = [1.0, 1.0, -0.5, -0.5, 1.5, 1.5, 0.75, 0.75]
    assert_array_almost_equal(xp, xp_expected)
    assert_array_almost_equal(yp, yp_expected)

    assert_almost_equal(f.avrg(), (1.0-0.5+0.5*1.5+1.5*0.75)/4.0, decimal=16)
    assert_almost_equal(f.abs_avrg(), (1.0+0.5+0.5*1.5+1.5*0.75)/4.0,
                        decimal=16)

    f1 = copy(f)
    x = [0.0, 0.75, 2.0, 2.5, 2.7, 4.0]
    y = [0.5, 1.0, -0.25, 0.0, 1.5]
    f2 = spk.PieceWiseConstFunc(x, y)
    f1.add(f2)
    x_expected = [0.0, 0.75, 1.0, 2.0, 2.5, 2.7, 4.0]
    y_expected = [1.5, 2.0, 0.5, 1.25, 0.75, 2.25]
    assert_array_almost_equal(f1.x, x_expected, decimal=16)
    assert_array_almost_equal(f1.y, y_expected, decimal=16)

    f2.add(f)
    assert_array_almost_equal(f2.x, x_expected, decimal=16)
    assert_array_almost_equal(f2.y, y_expected, decimal=16)
    
    f1.add(f2)
    # same x, but y doubled
    assert_array_almost_equal(f1.x, f2.x, decimal=16)
    assert_array_almost_equal(f1.y, 2*f2.y, decimal=16)


if __name__ == "__main__":
    test_pwc()
