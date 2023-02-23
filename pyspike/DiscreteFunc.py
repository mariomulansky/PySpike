# Class representing discrete functions.
# Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

from __future__ import absolute_import, print_function

import numpy as np
import collections.abc
import pyspike


##############################################################
# DiscreteFunc
##############################################################
class DiscreteFunc(object):
    """ A class representing values defined on a discrete set of points.
    """

    def __init__(self, x, y, multiplicity):
        """ Constructs the discrete function.

        :param x: array of length N defining the points at which the values are
                  defined.
        :param y: array of length N degining the values at the points x.
        :param multiplicity: array of length N defining the multiplicity of the
                             values.
        """
        # convert parameters to arrays, also ensures copying
        self.x = np.array(x)
        self.y = np.array(y)
        self.mp = np.array(multiplicity)

    def copy(self):
        """ Returns a copy of itself

        :rtype: :class:`DiscreteFunc`
        """
        return DiscreteFunc(self.x, self.y, self.mp)

    def almost_equal(self, other, decimal=14):
        """ Checks if the function is equal to another function up to `decimal`
        precision.

        :param other: another :class:`DiscreteFunc`
        :returns: True if the two functions are equal up to `decimal` decimals,
                  False otherwise
        :rtype: bool
        """
        eps = 10.0**(-decimal)
        return np.allclose(self.x, other.x, atol=eps, rtol=0.0) and \
            np.allclose(self.y, other.y, atol=eps, rtol=0.0) and \
            np.allclose(self.mp, other.mp, atol=eps, rtol=0.0)

    def get_plottable_data(self, averaging_window_size=0):
        """ Returns two arrays containing x- and y-coordinates for plotting
        the interval sequence. The optional parameter `averaging_window_size`
        determines the size of an averaging window to smoothen the profile. If
        this value is 0, no averaging is performed.

        :param averaging_window_size: size of the averaging window, default=0.
        :returns: (x_plot, y_plot) containing plottable data
        :rtype: pair of np.array

        Example::

            x, y = f.get_plottable_data()
            plt.plot(x, y, '-o', label="Discrete function")
        """

        if averaging_window_size > 0:
            # for the averaged profile we have to take the multiplicity into
            # account. values with higher multiplicity should be consider as if
            # they appeared several times. Hence we can not know how many
            # entries we have to consider to the left and right. Rather, we
            # will iterate until some wanted multiplicity is reached.

            # the first value in self.mp contains the number of averaged
            # profiles without any possible extra multiplicities
            # (by implementation)
            expected_mp = (averaging_window_size+1) * int(self.mp[0])
            y_plot = np.zeros_like(self.y)
            # compute the values in a loop, could be done in cython if required
            for i in range(len(y_plot)):

                if self.mp[i] >= expected_mp:
                    # the current value contains already all the wanted
                    # multiplicity
                    y_plot[i] = self.y[i]/self.mp[i]
                    continue

                # first look to the right
                y = self.y[i]
                mp_r = self.mp[i]
                j = i+1
                while j < len(y_plot):
                    if mp_r+self.mp[j] < expected_mp:
                        # if we still dont reach the required multiplicity
                        # we take the whole value
                        y += self.y[j]
                        mp_r += self.mp[j]
                    else:
                        # otherwise, just some fraction
                        y += self.y[j] * (expected_mp - mp_r)/self.mp[j]
                        mp_r += (expected_mp - mp_r)
                        break
                    j += 1

                # same story to the left
                mp_l = self.mp[i]
                j = i-1
                while j >= 0:
                    if mp_l+self.mp[j] < expected_mp:
                        y += self.y[j]
                        mp_l += self.mp[j]
                    else:
                        y += self.y[j] * (expected_mp - mp_l)/self.mp[j]
                        mp_l += (expected_mp - mp_l)
                        break
                    j -= 1
                y_plot[i] = y/(mp_l+mp_r-self.mp[i])
            return 1.0*self.x, y_plot

        else:  # k = 0

            return 1.0*self.x, 1.0*self.y/self.mp

    def integral(self, interval=None):
        """ Returns the integral over the given interval. For the discrete
        function, this amounts to two values: the sum over all values and the
        sum over all multiplicities.

        :param interval: integration interval given as a pair of floats, or a
                         sequence of pairs in case of multiple intervals, if
                         None the integral over the whole function is computed.
        :type interval: Pair, sequence of pairs, or None.
        :returns: the summed values and the summed multiplicity
        :rtype: pair of float
        """

        value = 0.0
        multiplicity = 0.0

        def get_indices(ival):
            """ Retuns the indeces surrounding the given interval"""
            start_ind = np.searchsorted(self.x, ival[0], side='right')
            end_ind = np.searchsorted(self.x, ival[1], side='left')
            assert start_ind > 0 and end_ind < len(self.x), \
                "Invalid averaging interval"
            return start_ind, end_ind

        if interval is None:
            # no interval given, integrate over the whole spike train
            # don't count the first value, which is zero by definition
            value = 1.0 * np.sum(self.y[1:-1])
            multiplicity = np.sum(self.mp[1:-1])
        else:
            # check if interval is as sequence
            assert isinstance(interval, collections.abc.Sequence), \
                "Invalid value for `interval`. None, Sequence or Tuple \
expected."
            # check if interval is a sequence of intervals
            if not isinstance(interval[0], collections.abc.Sequence):
                # find the indices corresponding to the interval
                start_ind, end_ind = get_indices(interval)
                value = np.sum(self.y[start_ind:end_ind])
                multiplicity = np.sum(self.mp[start_ind:end_ind])
            else:
                for ival in interval:
                    # find the indices corresponding to the interval
                    start_ind, end_ind = get_indices(ival)
                    value += np.sum(self.y[start_ind:end_ind])
                    multiplicity += np.sum(self.mp[start_ind:end_ind])
        return (value, multiplicity)

    def avrg(self, interval=None, normalize=True):
        """ Computes the average of the interval sequence:
        :math:`a = 1/N \\sum f_n` where N is the number of intervals.

        :param interval: averaging interval given as a pair of floats, a
                         sequence of pairs for averaging multiple intervals, or
                         None, if None the average over the whole function is
                         computed.
        :type interval: Pair, sequence of pairs, or None.
        :returns: the average a.
        :rtype: float
        """
        val, mp = self.integral(interval)
        if normalize:
            if mp > 0:
                return val/mp
            else:
                return 1.0
        else:
            return val

    def add(self, f):
        """ Adds another `DiscreteFunc` function to this function.
        Note: only functions defined on the same interval can be summed.

        :param f: :class:`DiscreteFunc` function to be added.
        :rtype: None
        """
        assert self.x[0] == f.x[0], "The functions have different intervals"
        assert self.x[-1] == f.x[-1], "The functions have different intervals"

        # cython version
        try:
            from .cython.cython_add import add_discrete_function_cython as \
                add_discrete_function_impl
        except ImportError:
            pyspike.NoCythonWarn()

            # use python backend
            from .cython.python_backend import add_discrete_function_python as \
                add_discrete_function_impl

        self.x, self.y, self.mp = \
            add_discrete_function_impl(self.x, self.y, self.mp,
                                       f.x, f.y, f.mp)

    def mul_scalar(self, fac):
        """ Multiplies the function with a scalar value

        :param fac: Value to multiply
        :type fac: double
        :rtype: None
        """
        self.y *= fac


def average_profile(profiles):
    """ Computes the average profile from the given ISI- or SPIKE-profiles.

    :param profiles: list of :class:`PieceWiseConstFunc` or
                     :class:`PieceWiseLinFunc` representing ISI- or
                     SPIKE-profiles to be averaged.
    :returns: the averages profile :math:`<S_{isi}>` or :math:`<S_{spike}>`.
    :rtype: :class:`PieceWiseConstFunc` or :class:`PieceWiseLinFunc`
    """
    assert len(profiles) > 1

    avrg_profile = profiles[0].copy()
    for i in range(1, len(profiles)):
        avrg_profile.add(profiles[i])
    avrg_profile.mul_scalar(1.0/len(profiles))  # normalize

    return avrg_profile
