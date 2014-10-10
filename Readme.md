# PySpike

PySpike is a Python library for numerical analysis of spike train similarity. 
Its core functionality is the implementation of the bivariate [ISI and SPIKE distance](http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony). 
Additionally, it allows to compute multi-variate spike train distances, averaging and general spike train processing.

All source codes are published under the liberal [MIT License](http://opensource.org/licenses/MIT).

## Requirements and Installation

To use PySpike you need Python installed with the following additional packages:

- numpy
- scipy
- matplotlib
- cython

In particular, make sure that [cython](http://www.cython.org) is configured properly and able to locate a C compiler.

To install PySpike, simply download the source, i.e. via git, and run the setup.py script:
    git clone ...
    cd PySpike
    python setup.py build_ext --inplace

## Loading spike trains


## Computing bi-variate distances


## Computing multi-variate distances


## Plotting


## Averaging