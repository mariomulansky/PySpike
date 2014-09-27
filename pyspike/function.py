""" function.py

Module containing classes representing piece-wise constant and piece-wise linear
functions.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

"""
from __future__ import print_function

import numpy as np

class PieceWiseConstFunc:
    """ A class representing a piece-wise constant function. """
    
    def __init__(self, x, y):
        """ Constructs the piece-wise const function.
        Params:
        - x: array of length N+1 defining the edges of the intervals of the pwc
        function.
        - y: array of length N defining the function values at the intervals.
        """
        self.x = np.array(x)
        self.y = np.array(y)

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
        Params:
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

class PieceWiseLinFunc:
    """ A class representing a piece-wise linear function. """
    
    def __init__(self, x, y1, y2):
        """ Constructs the piece-wise linear function.
        Params:
        - x: array of length N+1 defining the edges of the intervals of the pwc
        function.
        - y1: array of length N defining the function values at the left of the 
        intervals.
        - y2: array of length N defining the function values at the right of the 
        intervals.
        """
        self.x = x
        self.y1 = y1
        self.y2 = y2

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

