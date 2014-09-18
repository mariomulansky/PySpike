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
        self.x = x
        self.y = y

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
