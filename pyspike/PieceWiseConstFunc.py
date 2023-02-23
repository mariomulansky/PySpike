# Class representing piece-wise constant functions.
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import, print_function

import numpy as np
import collections.abc
import pyspike


##############################################################
# PieceWiseConstFunc
##############################################################
class PieceWiseConstFunc(object):
    """ A class representing a piece-wise constant function. """

    def __init__(self, x, y):
        """ Constructs the piece-wise const function.

        :param x: array of length N+1 defining the edges of the intervals of
                  the pwc function.
        :param y: array of length N defining the function values at the
                  intervals.
        """
        # convert parameters to arrays, also ensures copying
        self.x = np.array(x)
        self.y = np.array(y)

    def __call__(self, t):
        """ Returns the function value for the given time t. If t is a list of
        times, the corresponding list of values is returned.

        :param: time t, or list of times
        :returns: function value(s) at that time(s).
        """
        assert np.all(t >= self.x[0]) and np.all(t <= self.x[-1]), \
            "Invalid time: " + str(t)

        ind = np.searchsorted(self.x, t, side='right')

        if isinstance(t, collections.abc.Sequence):
            # t is a sequence of values
            # correct the cases t == x[0], t == x[-1]
            ind[ind == 0] = 1
            ind[ind == len(self.x)] = len(self.x)-1
            value = self.y[ind-1]
            # correct the values at exact spike times: there the value should
            # be the at half of the step
            # obtain the 'left' side indices for t
            ind_l = np.searchsorted(self.x, t, side='left')
            # if left and right side indices differ, the time t has to appear
            # in self.x
            ind_at_spike = np.logical_and(np.logical_and(ind != ind_l,
                                                         ind > 1),
                                          ind < len(self.x))
            # get the corresponding indices for the resulting value array
            val_ind = np.arange(len(ind))[ind_at_spike]
            # and for the arrays self.x, y1, y2
            xy_ind = ind[ind_at_spike]
            # use the middle of the left and right ISI value
            value[val_ind] = 0.5 * (self.y[xy_ind-1] + self.y[xy_ind-2])
            return value
        else:  # t is a single value
            # specific check for interval edges
            if t == self.x[0]:
                return self.y[0]
            if t == self.x[-1]:
                return self.y[-1]
            # check if we are on any other exact spike time
            if sum(self.x == t) > 0:
                # use the middle of the left and right ISI value
                return 0.5 * (self.y[ind-1] + self.y[ind-2])
            return self.y[ind-1]

    def copy(self):
        """ Returns a copy of itself

        :rtype: :class:`PieceWiseConstFunc`
        """
        return PieceWiseConstFunc(self.x, self.y)

    def almost_equal(self, other, decimal=14):
        """ Checks if the function is equal to another function up to `decimal`
        precision.

        :param other: another :class:`PieceWiseConstFunc`
        :returns: True if the two functions are equal up to `decimal` decimals,
                  False otherwise
        :rtype: bool
        """
        eps = 10.0**(-decimal)
        return np.allclose(self.x, other.x, atol=eps, rtol=0.0) and \
            np.allclose(self.y, other.y, atol=eps, rtol=0.0)

    def get_plottable_data(self):
        """ Returns two arrays containing x- and y-coordinates for immeditate
        plotting of the piece-wise function.

        :returns: (x_plot, y_plot) containing plottable data
        :rtype: pair of np.array

        Example::

            x, y = f.get_plottable_data()
            plt.plot(x, y, '-o', label="Piece-wise const function")
        """

        x_plot = np.empty(2*len(self.x)-2)
        x_plot[0] = self.x[0]
        x_plot[1::2] = self.x[1:]
        x_plot[2::2] = self.x[1:-1]
        y_plot = np.empty(2*len(self.y))
        y_plot[::2] = self.y
        y_plot[1::2] = self.y

        return x_plot, y_plot

    def integral(self, interval=None):
        """ Returns the integral over the given interval.

        :param interval: integration interval given as a pair of floats, if
                         None the integral over the whole function is computed.
        :type interval: Pair of floats or None.
        :returns: the integral
        :rtype: float
        """
        if interval is None:
            # no interval given, integrate over the whole spike train
            a = np.sum((self.x[1:]-self.x[:-1]) * self.y)
        else:
            if interval[0]>interval[1]:
                raise ValueError("Invalid averaging interval: interval[0]>=interval[1]")
            if interval[0]<self.x[0]:
                raise ValueError("Invalid averaging interval: interval[0]<self.x[0]")
            if interval[1]>self.x[-1]:
                raise ValueError("Invalid averaging interval: interval[0]<self.x[-1]")
            # find the indices corresponding to the interval
            start_ind = np.searchsorted(self.x, interval[0], side='right')
            end_ind = np.searchsorted(self.x, interval[1], side='left')-1
            if start_ind > end_ind:
                # contribution from between two closest edges
                a = (self.x[start_ind]-self.x[end_ind]) * self.y[end_ind]
                # minus the part that is not within the interval
                a -= ((interval[0]-self.x[end_ind])+(self.x[start_ind]-interval[1])) * self.y[end_ind]
            else:
                assert start_ind > 0 and end_ind < len(self.x), \
                    "Invalid averaging interval"
                # first the contribution from between the indices
                a = np.sum((self.x[start_ind+1:end_ind+1] -
                            self.x[start_ind:end_ind]) *
                           self.y[start_ind:end_ind])
                # correction from start to first index
                a += (self.x[start_ind]-interval[0]) * self.y[start_ind-1]
                # correction from last index to end
                a += (interval[1]-self.x[end_ind]) * self.y[end_ind]
        return a

    def avrg(self, interval=None):
        """ Computes the average of the piece-wise const function:
        :math:`a = 1/T \int_0^T f(x) dx` where T is the length of the interval.

        :param interval: averaging interval given as a pair of floats, a
                         sequence of pairs for averaging multiple intervals, or
                         None, if None the average over the whole function is
                         computed.
        :type interval: Pair, sequence of pairs, or None.
        :returns: the average a.
        :rtype: float
        """
        if interval is None:
            # no interval given, average over the whole spike train
            return self.integral() / (self.x[-1]-self.x[0])

        # check if interval is as sequence
        assert isinstance(interval, collections.abc.Sequence), \
            "Invalid value for `interval`. None, Sequence or Tuple expected."
        # check if interval is a sequence of intervals
        if not isinstance(interval[0], collections.abc.Sequence):
            # just one interval
            a = self.integral(interval) / (interval[1]-interval[0])
        else:
            # several intervals
            a = 0.0
            int_length = 0.0
            for ival in interval:
                a += self.integral(ival)
                int_length += ival[1] - ival[0]
            a /= int_length
        return a

    def add(self, f):
        """ Adds another PieceWiseConst function to this function.
        Note: only functions defined on the same interval can be summed.

        :param f: :class:`PieceWiseConstFunc` function to be added.
        :rtype: None
        """
        assert self.x[0] == f.x[0], "The functions have different intervals"
        assert self.x[-1] == f.x[-1], "The functions have different intervals"

        # cython version
        try:
            from .cython.cython_add import add_piece_wise_const_cython as \
                add_piece_wise_const_impl
        except ImportError:
            pyspike.NoCythonWarn()

            # use python backend
            from .cython.python_backend import add_piece_wise_const_python as \
                add_piece_wise_const_impl

        self.x, self.y = add_piece_wise_const_impl(self.x, self.y, f.x, f.y)

    def mul_scalar(self, fac):
        """ Multiplies the function with a scalar value

        :param fac: Value to multiply
        :type fac: double
        :rtype: None
        """
        self.y *= fac
