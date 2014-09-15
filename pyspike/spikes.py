""" spikes.py

Module containing several function to load and transform spike trains

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

import numpy as np

def spike_train_from_string(s, sep=' '):
    """ Converts a string of times into a SpikeTrain object.
    Params:
    - s: the string with (ordered) spike times
    - sep: The separator between the time numbers.
    Returns:
    - array of spike times
    """
    return np.fromstring(s, sep=sep)
