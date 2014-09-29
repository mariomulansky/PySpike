""" function.py

Module containing classes representing piece-wise constant and piece-wise linear
functions.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

"""
from __future__ import print_function

import numpy as np


##############################################################
# PieceWiseConstFunc
##############################################################
class PieceWiseConstFunc:
    """ A class representing a piece-wise constant function. """
    
    def __init__(self, x, y):
        """ Constructs the piece-wise const function.
        Args:
        - x: array of length N+1 defining the edges of the intervals of the pwc
        function.
        - y: array of length N defining the function values at the intervals.
        """
        self.x = np.array(x)
        self.y = np.array(y)

    def almost_equal(self, other, decimal=14):
        """ Checks if the function is equal to another function up to `decimal`
        precision.
        Args:
        - other: another PieceWiseConstFunc object
        Returns:
        True if the two functions are equal up to `decimal` decimals, 
        False otherwise
        """
        eps = 10.0**(-decimal)
        return np.allclose(self.x, other.x, atol=eps, rtol=0.0) and \
            np.allclose(self.y, other.y, atol=eps, rtol=0.0)

    def get_plottable_data(self):
        """ Returns two arrays containing x- and y-coordinates for immeditate
        plotting of the piece-wise function.
        """

        x_plot = np.empty(2*len(self.x)-2)
        x_plot[0] = self.x[0]
        x_plot[1::2] = self.x[1:]
        x_plot[2::2] = self.x[1:-1]
        y_plot = np.empty(2*len(self.y))
        y_plot[::2] = self.y
        y_plot[1::2] = self.y

        return x_plot, y_plot

    def avrg(self):
        """ Computes the average of the piece-wise const function:
        a = 1/T int f(x) dx where T is the length of the interval.
        Returns:
        - the average a. 
        """
        return np.sum((self.x[1:]-self.x[:-1]) * self.y) / \
            (self.x[-1]-self.x[0])

    def abs_avrg(self):
        """ Computes the average of the abs value of the piece-wise const 
        function:
        a = 1/T int |f(x)| dx where T is the length of the interval.
        Returns:
        - the average a. 
        """
        return np.sum((self.x[1:]-self.x[:-1]) * np.abs(self.y)) / \
            (self.x[-1]-self.x[0])

    def add(self, f):
        """ Adds another PieceWiseConst function to this function. 
        Note: only functions defined on the same interval can be summed.
        Args:
        - f: PieceWiseConst function to be added.
        """
        assert self.x[0] == f.x[0], "The functions have different intervals"
        assert self.x[-1] == f.x[-1], "The functions have different intervals"
        x_new = np.empty(len(self.x) + len(f.x))
        y_new = np.empty(len(x_new)-1)
        x_new[0] = self.x[0]
        y_new[0] = self.y[0] + f.y[0]
        index1 = 0
        index2 = 0
        index = 0
        while (index1+1 < len(self.y)) and (index2+1 < len(f.y)):
            index += 1
            # print(index1+1, self.x[index1+1], self.y[index1+1], x_new[index])
            if self.x[index1+1] < f.x[index2+1]:
                index1 += 1
                x_new[index] = self.x[index1]
            elif self.x[index1+1] > f.x[index2+1]:
                index2 += 1
                x_new[index] = f.x[index2]
            else: # self.x[index1+1] == f.x[index2+1]:
                index1 += 1
                index2 += 1
                x_new[index] = self.x[index1]
            y_new[index] = self.y[index1] + f.y[index2]
        # one array reached the end -> copy the contents of the other to the end
        if index1+1 < len(self.y):
            x_new[index+1:index+1+len(self.x)-index1-1] = self.x[index1+1:]
            y_new[index+1:index+1+len(self.y)-index1-1] = self.y[index1+1:] + \
                                                          f.y[-1]
            index += len(self.x)-index1-2
        elif index2+1 < len(f.y):
            x_new[index+1:index+1+len(f.x)-index2-1] = f.x[index2+1:]
            y_new[index+1:index+1+len(f.y)-index2-1] = f.y[index2+1:] + \
                                                       self.y[-1]
            index += len(f.x)-index2-2
        else: # both arrays reached the end simultaneously
            # only the last x-value missing
            x_new[index+1] = self.x[-1]
        # the last value is again the end of the interval
        # x_new[index+1] = self.x[-1]
        # only use the data that was actually filled
        self.x = x_new[:index+2]
        self.y = y_new[:index+1]

    def mul_scalar(self, fac):
        """ Multiplies the function with a scalar value
        Args:
        - fac: Value to multiply
        """
        self.y *= fac


##############################################################
# PieceWiseLinFunc
##############################################################
class PieceWiseLinFunc:
    """ A class representing a piece-wise linear function. """
    
    def __init__(self, x, y1, y2):
        """ Constructs the piece-wise linear function.
        Args:
        - x: array of length N+1 defining the edges of the intervals of the pwc
        function.
        - y1: array of length N defining the function values at the left of the 
        intervals.
        - y2: array of length N defining the function values at the right of the 
        intervals.
        """
        self.x = np.array(x)
        self.y1 = np.array(y1)
        self.y2 = np.array(y2)

    def almost_equal(self, other, decimal=14):
        """ Checks if the function is equal to another function up to `decimal`
        precision.
        Args:
        - other: another PieceWiseLinFunc object
        Returns:
        True if the two functions are equal up to `decimal` decimals, 
        False otherwise
        """
        eps = 10.0**(-decimal)
        return np.allclose(self.x, other.x, atol=eps, rtol=0.0) and \
            np.allclose(self.y1, other.y1, atol=eps, rtol=0.0) and \
            np.allclose(self.y2, other.y2, atol=eps, rtol=0.0)

    def get_plottable_data(self):
        """ Returns two arrays containing x- and y-coordinates for immeditate
        plotting of the piece-wise function.
        """
        x_plot = np.empty(2*len(self.x)-2)
        x_plot[0] = self.x[0]
        x_plot[1::2] = self.x[1:]
        x_plot[2::2] = self.x[1:-1]
        y_plot = np.empty_like(x_plot)
        y_plot[0::2] = self.y1
        y_plot[1::2] = self.y2
        return x_plot, y_plot

    def avrg(self):
        """ Computes the average of the piece-wise linear function:
        a = 1/T int f(x) dx where T is the length of the interval.
        Returns:
        - the average a. 
        """
        return np.sum((self.x[1:]-self.x[:-1]) * 0.5*(self.y1+self.y2)) / \
            (self.x[-1]-self.x[0])

    def abs_avrg(self):
        """ Computes the absolute average of the piece-wise linear function:
        a = 1/T int |f(x)| dx where T is the length of the interval.
        Returns:
        - the average a. 
        """
        return np.sum((self.x[1:]-self.x[:-1]) * 0.5 *
                      (np.abs(self.y1)+np.abs(self.y2)))/(self.x[-1]-self.x[0])

    def add(self, f):
        """ Adds another PieceWiseLin function to this function. 
        Note: only functions defined on the same interval can be summed.
        Args:
        - f: PieceWiseLin function to be added.
        """
        assert self.x[0] == f.x[0], "The functions have different intervals"
        assert self.x[-1] == f.x[-1], "The functions have different intervals"
        x_new = np.empty(len(self.x) + len(f.x))
        y1_new = np.empty(len(x_new)-1)
        y2_new = np.empty_like(y1_new)
        x_new[0] = self.x[0]
        y1_new[0] = self.y1[0] + f.y1[0]
        index1 = 0 # index for self
        index2 = 0 # index for f
        index = 0  # index for new
        while (index1+1 < len(self.y1)) and (index2+1 < len(f.y1)):
            # print(index1+1, self.x[index1+1], self.y[index1+1], x_new[index])
            if self.x[index1+1] < f.x[index2+1]:
                # first compute the end value of the previous interval
                # linear interpolation of the interval
                y = f.y1[index2] + (f.y2[index2]-f.y1[index2]) * \
                    (self.x[index1+1]-f.x[index2]) / (f.x[index2+1]-f.x[index2])
                y2_new[index] = self.y2[index1] + y
                index1 += 1
                index += 1
                x_new[index] = self.x[index1]
                # and the starting value for the next interval
                y1_new[index] = self.y1[index1] + y
            elif self.x[index1+1] > f.x[index2+1]:
                # first compute the end value of the previous interval
                # linear interpolation of the interval
                y = self.y1[index1] + (self.y2[index1]-self.y1[index1]) * \
                    (f.x[index2+1]-self.x[index1]) / \
                    (self.x[index1+1]-self.x[index1])
                y2_new[index] = f.y2[index2] + y
                index2 += 1
                index += 1
                x_new[index] = f.x[index2]
                # and the starting value for the next interval
                y1_new[index] = f.y1[index2] + y
            else: # self.x[index1+1] == f.x[index2+1]:
                y2_new[index] = self.y2[index1] + f.y2[index2]
                index1 += 1
                index2 += 1
                index += 1
                x_new[index] = self.x[index1]
                y1_new[index] = self.y1[index1] + f.y1[index2]
        # one array reached the end -> copy the contents of the other to the end
        if index1+1 < len(self.y1):
            # compute the linear interpolations values
            y = f.y1[index2] + (f.y2[index2]-f.y1[index2]) * \
                (self.x[index1+1:-1]-f.x[index2]) / (f.x[index2+1]-f.x[index2])
            x_new[index+1:index+1+len(self.x)-index1-1] = self.x[index1+1:]
            y1_new[index+1:index+1+len(self.y1)-index1-1] = self.y1[index1+1:]+y
            y2_new[index:index+len(self.y2)-index1-1] = self.y2[index1:-1] + y
            index += len(self.x)-index1-2
        elif index2+1 < len(f.y1):
            # compute the linear interpolations values
            y = self.y1[index1] + (self.y2[index1]-self.y1[index1]) * \
                (f.x[index2+1:-1]-self.x[index1]) / \
                (self.x[index1+1]-self.x[index1])
            x_new[index+1:index+1+len(f.x)-index2-1] = f.x[index2+1:]
            y1_new[index+1:index+1+len(f.y1)-index2-1] = f.y1[index2+1:] + y
            y2_new[index:index+len(f.y2)-index2-1] = f.y2[index2:-1] + y
            index += len(f.x)-index2-2
        else: # both arrays reached the end simultaneously
            # only the last x-value missing
            x_new[index+1] = self.x[-1]
        # finally, the end value for the last interval
        y2_new[index] = self.y2[-1]+f.y2[-1]
        # only use the data that was actually filled
        self.x = x_new[:index+2]
        self.y1 = y1_new[:index+1]
        self.y2 = y2_new[:index+1]

    def mul_scalar(self, fac):
        """ Multiplies the function with a scalar value
        Args:
        - fac: Value to multiply
        """
        self.y1 *= fac
        self.y2 *= fac
