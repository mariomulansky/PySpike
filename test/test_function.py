""" test_function.py

Tests the PieceWiseConst and PieceWiseLinear functions

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import print_function
import numpy as np
from copy import copy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, \
    assert_array_equal, assert_array_almost_equal

import pyspike as spk


def test_pwc():
    # some random data
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [1.0, -0.5, 1.5, 0.75]
    f = spk.PieceWiseConstFunc(x, y)

    # function values
    assert_allclose(f(0.0), 1.0)
    assert_allclose(f(0.5), 1.0)
    assert_allclose(f(1.0), 0.25)
    assert_allclose(f(2.0), 0.5)
    assert_allclose(f(2.25), 1.5)
    assert_allclose(f(2.5), 2.25/2)
    assert_allclose(f(3.5), 0.75)
    assert_allclose(f(4.0), 0.75)

    assert_array_equal(f([0.0, 0.5, 1.0, 2.0, 2.25, 2.5, 3.5, 4.0]),
                       [1.0, 1.0, 0.25, 0.5, 1.5, 2.25/2, 0.75, 0.75])

    xp, yp = f.get_plottable_data()

    xp_expected = [0.0, 1.0, 1.0, 2.0, 2.0, 2.5, 2.5, 4.0]
    yp_expected = [1.0, 1.0, -0.5, -0.5, 1.5, 1.5, 0.75, 0.75]
    assert_array_almost_equal(xp, xp_expected, decimal=16)
    assert_array_almost_equal(yp, yp_expected, decimal=16)

    assert_almost_equal(f.avrg(), (1.0-0.5+0.5*1.5+1.5*0.75)/4.0, decimal=16)

    # interval averaging
    a = f.avrg([0.5, 3.5])
    assert_almost_equal(a, (0.5-0.5+0.5*1.5+1.0*0.75)/3.0, decimal=16)
    a = f.avrg([1.5, 3.5])
    assert_almost_equal(a, (-0.5*0.5+0.5*1.5+1.0*0.75)/2.0, decimal=16)
    a = f.avrg([1.0, 2.0])
    assert_almost_equal(a, (1.0*-0.5)/1.0, decimal=16)
    a = f.avrg([1.0, 3.5])
    assert_almost_equal(a, (-0.5*1.0+0.5*1.5+1.0*0.75)/2.5, decimal=16)
    a = f.avrg([1.0, 4.0])
    assert_almost_equal(a, (-0.5*1.0+0.5*1.5+1.5*0.75)/3.0, decimal=16)
    a = f.avrg([0.0, 2.2])
    assert_almost_equal(a, (1.0*1.0-0.5*1.0+0.2*1.5)/2.2, decimal=15)

    # averaging over multiple intervals
    a = f.avrg([(0.5, 1.5), (1.5, 3.5)])
    assert_almost_equal(a, (0.5-0.5+0.5*1.5+1.0*0.75)/3.0, decimal=16)

    # averaging over multiple intervals
    a = f.avrg([(0.5, 1.5), (2.2, 3.5)])
    assert_almost_equal(a, (0.5*1.0-0.5*0.5+0.3*1.5+1.0*0.75)/2.3, decimal=15)


def test_pwc_add():
    # some random data
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [1.0, -0.5, 1.5, 0.75]
    f = spk.PieceWiseConstFunc(x, y)

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


def test_pwc_mul():
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [1.0, -0.5, 1.5, 0.75]
    f = spk.PieceWiseConstFunc(x, y)

    f.mul_scalar(1.5)
    assert_array_almost_equal(f.x, x, decimal=16)
    assert_array_almost_equal(f.y, 1.5*np.array(y), decimal=16)
    f.mul_scalar(1.0/5.0)
    assert_array_almost_equal(f.y, 1.5/5.0*np.array(y), decimal=16)


def test_pwc_avrg():
    # some random data
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [1.0, -0.5, 1.5, 0.75]
    f1 = spk.PieceWiseConstFunc(x, y)

    x = [0.0, 0.75, 2.0, 2.5, 2.7, 4.0]
    y = [0.5, 1.0, -0.25, 0.0, 1.5]
    f2 = spk.PieceWiseConstFunc(x, y)

    f1.add(f2)
    f1.mul_scalar(0.5)
    x_expected = [0.0, 0.75, 1.0, 2.0, 2.5, 2.7, 4.0]
    y_expected = [0.75, 1.0, 0.25, 0.625, 0.375, 1.125]
    assert_array_almost_equal(f1.x, x_expected, decimal=16)
    assert_array_almost_equal(f1.y, y_expected, decimal=16)

def test_pwc_integral():
    # some random data
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [1.0, -0.5, 1.5, 0.75]
    f1 = spk.PieceWiseConstFunc(x, y)

    # test full interval
    full = 1.0*1.0 + 1.0*-0.5 + 0.5*1.5 + 1.5*0.75;
    assert_allclose(f1.integral(), full)
    assert_allclose(f1.integral((np.min(x),np.max(x))), full)
    # test part interval, spanning an edge
    assert_allclose(f1.integral((0.5,1.5)), 0.5*1.0 + 0.5*-0.5)
    # test part interval, just over two edges
    assert_almost_equal(f1.integral((1.0-1e-16,2+1e-16)), 1.0*-0.5, decimal=14)
    # test part interval, between two edges
    assert_allclose(f1.integral((1.0,2.0)), 1.0*-0.5)
    assert_allclose(f1.integral((1.2,1.7)), (1.7-1.2)*-0.5)
    # test part interval, start to before and after edge
    assert_allclose(f1.integral((0.0,0.7)), 0.7*1.0)
    assert_allclose(f1.integral((0.0,1.1)), 1.0*1.0+0.1*-0.5)
    # test part interval, before and after edge till end
    assert_allclose(f1.integral((2.6,4.0)), (4.0-2.6)*0.75)
    assert_allclose(f1.integral((2.4,4.0)), (2.5-2.4)*1.5+(4-2.5)*0.75)

def test_pwc_integral_bad_bounds_inv():
    with pytest.raises(ValueError):
        # some random data
        x = [0.0, 1.0, 2.0, 2.5, 4.0]
        y = [1.0, -0.5, 1.5, 0.75]
        f1 = spk.PieceWiseConstFunc(x, y)
        f1.integral((3,2))

def test_pwc_integral_bad_bounds_oob_1():
    with pytest.raises(ValueError):
        # some random data
        x = [0.0, 1.0, 2.0, 2.5, 4.0]
        y = [1.0, -0.5, 1.5, 0.75]
        f1 = spk.PieceWiseConstFunc(x, y)
        f1.integral((1,6))

def test_pwc_integral_bad_bounds_oob_2():
    with pytest.raises(ValueError):
        # some random data
        x = [0.0, 1.0, 2.0, 2.5, 4.0]
        y = [1.0, -0.5, 1.5, 0.75]
        f1 = spk.PieceWiseConstFunc(x, y)
        f1.integral((-1,3))

def test_pwl():
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y1 = [1.0, -0.5, 1.5, 0.75]
    y2 = [1.5, -0.4, 1.5, 0.25]
    f = spk.PieceWiseLinFunc(x, y1, y2)

    # function values
    assert_allclose(f(0.0), 1.0)
    assert_allclose(f(0.5), 1.25)
    assert_allclose(f(1.0), 0.5)
    assert_allclose(f(2.0), 1.1/2)
    assert_allclose(f(2.25), 1.5)
    assert_allclose(f(2.5), 2.25/2)
    assert_allclose(f(3.5), 0.75-0.5*1.0/1.5)
    assert_allclose(f(4.0), 0.25)

    assert_array_equal(f([0.0, 0.5, 1.0, 2.0, 2.25, 2.5, 3.5, 4.0]),
                       [1.0, 1.25, 0.5, 0.55, 1.5, 2.25/2, 0.75-0.5/1.5, 0.25])

    xp, yp = f.get_plottable_data()

    xp_expected = [0.0, 1.0, 1.0, 2.0, 2.0, 2.5, 2.5, 4.0]
    yp_expected = [1.0, 1.5, -0.5, -0.4, 1.5, 1.5, 0.75, 0.25]
    assert_array_almost_equal(xp, xp_expected, decimal=16)
    assert_array_almost_equal(yp, yp_expected, decimal=16)

    avrg_expected = (1.25 - 0.45 + 0.75 + 1.5*0.5) / 4.0
    assert_almost_equal(f.avrg(), avrg_expected, decimal=16)

    # interval averaging
    a = f.avrg([0.5, 2.5])
    assert_almost_equal(a, (1.375*0.5 - 0.45 + 0.75)/2.0, decimal=16)
    a = f.avrg([1.5, 3.5])
    assert_almost_equal(a, (-0.425*0.5 + 0.75 + (0.75+0.75-0.5/1.5)/2) / 2.0,
                        decimal=16)
    a = f.avrg((1.0, 3.5))
    assert_almost_equal(a, (-0.45 + 0.75 + (0.75+0.75-0.5/1.5)/2) / 2.5,
                        decimal=16)
    a = f.avrg([1.0, 4.0])
    assert_almost_equal(a, (-0.45 + 0.75 + 1.5*0.5) / 3.0, decimal=16)

    # interval between support points
    a = f.avrg([1.1, 1.5])
    assert_almost_equal(a, (-0.5+0.1*0.1 - 0.45) * 0.5, decimal=14)

    # starting at a support point
    a = f.avrg([1.0, 1.5])
    assert_almost_equal(a, (-0.5 - 0.45) * 0.5, decimal=14)

    # start and end at support point
    a = f.avrg([1.0, 2.0])
    assert_almost_equal(a, (-0.5 - 0.4) * 0.5, decimal=14)
    
    # averaging over multiple intervals
    a = f.avrg([(0.5, 1.5), (1.5, 2.5)])
    assert_almost_equal(a, (1.375*0.5 - 0.45 + 0.75)/2.0, decimal=16)


def test_pwl_add():
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y1 = [1.0, -0.5, 1.5, 0.75]
    y2 = [1.5, -0.4, 1.5, 0.25]
    f = spk.PieceWiseLinFunc(x, y1, y2)

    f1 = copy(f)
    x = [0.0, 0.75, 2.0, 2.5, 2.7, 4.0]
    y1 = [0.5, 1.0, -0.25, 0.0, 1.5]
    y2 = [0.8, 0.2, -1.0, 0.0, 2.0]
    f2 = spk.PieceWiseLinFunc(x, y1, y2)
    f1.add(f2)
    x_expected = [0.0, 0.75, 1.0, 2.0, 2.5, 2.7, 4.0]
    y1_expected = [1.5, 1.0+1.0+0.5*0.75, -0.5+1.0-0.8*0.25/1.25, 1.5-0.25,
                   0.75, 1.5+0.75-0.5*0.2/1.5]
    y2_expected = [0.8+1.0+0.5*0.75, 1.5+1.0-0.8*0.25/1.25, -0.4+0.2, 1.5-1.0,
                   0.75-0.5*0.2/1.5, 2.25]
    assert_array_almost_equal(f1.x, x_expected, decimal=16)
    assert_array_almost_equal(f1.y1, y1_expected, decimal=16)
    assert_array_almost_equal(f1.y2, y2_expected, decimal=16)

    f2.add(f)
    assert_array_almost_equal(f2.x, x_expected, decimal=16)
    assert_array_almost_equal(f2.y1, y1_expected, decimal=16)
    assert_array_almost_equal(f2.y2, y2_expected, decimal=16)

    f1.add(f2)
    # same x, but y doubled
    assert_array_almost_equal(f1.x, f2.x, decimal=16)
    assert_array_almost_equal(f1.y1, 2*f2.y1, decimal=16)
    assert_array_almost_equal(f1.y2, 2*f2.y2, decimal=16)


def test_pwl_mul():
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y1 = [1.0, -0.5, 1.5, 0.75]
    y2 = [1.5, -0.4, 1.5, 0.25]
    f = spk.PieceWiseLinFunc(x, y1, y2)

    f.mul_scalar(1.5)
    assert_array_almost_equal(f.x, x, decimal=16)
    assert_array_almost_equal(f.y1, 1.5*np.array(y1), decimal=16)
    assert_array_almost_equal(f.y2, 1.5*np.array(y2), decimal=16)
    f.mul_scalar(1.0/5.0)
    assert_array_almost_equal(f.y1, 1.5/5.0*np.array(y1), decimal=16)
    assert_array_almost_equal(f.y2, 1.5/5.0*np.array(y2), decimal=16)


def test_pwl_avrg():
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y1 = [1.0, -0.5, 1.5, 0.75]
    y2 = [1.5, -0.4, 1.5, 0.25]
    f1 = spk.PieceWiseLinFunc(x, y1, y2)

    x = [0.0, 0.75, 2.0, 2.5, 2.7, 4.0]
    y1 = [0.5, 1.0, -0.25, 0.0, 1.5]
    y2 = [0.8, 0.2, -1.0, 0.0, 2.0]
    f2 = spk.PieceWiseLinFunc(x, y1, y2)

    x_expected = [0.0, 0.75, 1.0, 2.0, 2.5, 2.7, 4.0]
    y1_expected = np.array([1.5, 1.0+1.0+0.5*0.75, -0.5+1.0-0.8*0.25/1.25,
                            1.5-0.25, 0.75, 1.5+0.75-0.5*0.2/1.5]) / 2
    y2_expected = np.array([0.8+1.0+0.5*0.75, 1.5+1.0-0.8*0.25/1.25, -0.4+0.2,
                            1.5-1.0, 0.75-0.5*0.2/1.5, 2.25]) / 2

    f1.add(f2)
    f1.mul_scalar(0.5)

    assert_array_almost_equal(f1.x, x_expected, decimal=16)
    assert_array_almost_equal(f1.y1, y1_expected, decimal=16)
    assert_array_almost_equal(f1.y2, y2_expected, decimal=16)


def test_df():
    # testing discrete function
    x = [0.0, 1.0, 2.0, 2.5, 4.0]
    y = [0.0, 1.0, 1.0, 0.0, 1.0]
    mp = [1.0, 2.0, 1.0, 2.0, 1.0]
    f = spk.DiscreteFunc(x, y, mp)
    xp, yp = f.get_plottable_data()

    xp_expected = [0.0, 1.0, 2.0, 2.5, 4.0]
    yp_expected = [0.0, 0.5, 1.0, 0.0, 1.0]
    assert_array_almost_equal(xp, xp_expected, decimal=16)
    assert_array_almost_equal(yp, yp_expected, decimal=16)

    assert_almost_equal(f.avrg(), 2.0/5.0, decimal=16)

    # interval averaging
    a = f.avrg([0.5, 2.4])
    assert_almost_equal(a, 2.0/3.0, decimal=16)
    a = f.avrg([1.5, 3.5])
    assert_almost_equal(a, 1.0/3.0, decimal=16)
    a = f.avrg((0.9, 3.5))
    assert_almost_equal(a, 2.0/5.0, decimal=16)
    a = f.avrg([1.1, 4.0])
    assert_almost_equal(a, 1.0/3.0, decimal=16)

    # averaging over multiple intervals
    a = f.avrg([(0.5, 1.5), (1.5, 2.6)])
    assert_almost_equal(a, 2.0/5.0, decimal=16)


if __name__ == "__main__":
    test_pwc()
    test_pwc_add()
    test_pwc_mul()
    test_pwc_avrg()
    test_pwl()
    test_pwl_add()
    test_pwl_mul()
    test_pwl_avrg()
    test_df()

