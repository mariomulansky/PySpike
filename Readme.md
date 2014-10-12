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
- nosetests (for running the tests)

In particular, make sure that [cython](http://www.cython.org) is configured properly and able to locate a C compiler.

To install PySpike, simply download the source, i.e. via git clone, and run the setup.py script:

    git clone https://github.com/mariomulansky/PySpike.git
    cd PySpike
    python setup.py build_ext --inplace

Then you can run the tests using the `nosetests` test framework:

    cd test
    nosetests

Finally, you should make the installation folder known to Python to be able to import pyspike in your own projects.
Therefore, add your `/path/to/PySpike` to the `$PYTHONPATH` environment variable.

## Spike trains

In PySpike, spike trains are represented by one-dimensional numpy arrays containing the sequence of spike times as double values.
The following code creates such a spike train with some arbitrary spike times:
    
    import numpy as np

    spike_train = np.array([0.1, 0.3, 0.45, 0.6, 0.9])

Typically, spike train data is loaded into PySpike from data files.
The most straight-forward data files are text files where each line represents one spike train given as an sequence of spike times.
An exemplary file with several spike trains is [PySpike_testdata.txt](https://github.com/mariomulansky/PySpike/blob/master/examples/PySpike_testdata.txt).
To quickly obtain spike trains from such files, PySpike provides the function `load_spike_trains_from_txt`.

    import numpy as np
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 
                                                  time_interval=(0,4000))

This function expects the name of the datafile as first parameter, and additionally the time intervall of the spike train measurement can be provided as a pair of start- and end-time values.
If the time interval is provided (`time_interval is not None`), auxiliary spikes at the start- and end-time of the interval are added to the spike trains.
Furthermore, the spike trains are ordered via `np.sort`.
As result, `load_spike_trains_from_txt` returns a *list of arrays* containing the spike trains in the text file.


## Computing bi-variate distances

----------------------
**Important note:**

>Spike trains are expected to be *ordered sequences*! 
>For performance reasons, the PySpike distance function do not check if the spike trains provided are indeed ordered.
>Make sure that all your spike trains are ordered.
>If in doubt, use `spike_train = np.sort(spike_train)` to obtain a correctly ordered spike train.

----------------------


## Computing multi-variate distances


## Plotting


## Averaging
