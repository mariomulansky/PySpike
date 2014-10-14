# PySpike

PySpike is a Python library for the numerical analysis of spike train similarity. 
Its core functionality is the implementation of the bivariate [ISI](http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony#ISI-distance) [1] and [SPIKE](http://www.scholarpedia.org/article/SPIKE-distance) [2] distance. 
Additionally, it provides functions to compute multi-variate SPIKE and ISI distances, as well as averaging and general spike train processing.
All computation intensive parts are implemented in C via [cython](http://www.cython.org) to reach a competitive performance (factor 100-200 over plain Python).

All source codes are published under the [BSD License](http://opensource.org/licenses/BSD-2-Clause).

>[1] Kreuz T, Haas JS, Morelli A, Abarbanel HDI, Politi A, *Measuring spike train synchrony.* J Neurosci Methods 165, 151 (2007)

>[2] Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F, *Monitoring spike train synchrony.* J Neurophysiol 109, 1457 (2013)

## Requirements and Installation

To use PySpike you need Python installed with the following additional packages:

- numpy
- scipy
- matplotlib
- cython
- nosetests (for running the tests)

In particular, make sure that [cython](http://www.cython.org) is configured properly and able to locate a C compiler.

To install PySpike, simply download the source, e.g. from Github, and run the `setup.py` script:

    git clone https://github.com/mariomulansky/PySpike.git
    cd PySpike
    python setup.py build_ext --inplace

Then you can run the tests using the `nosetests` test framework:

    cd test
    nosetests

Finally, you should make PySpike's installation folder known to Python to be able to import pyspike in your own projects.
Therefore, add your `/path/to/PySpike` to the `$PYTHONPATH` environment variable.

## Spike trains

In PySpike, spike trains are represented by one-dimensional numpy arrays containing the sequence of spike times as double values.
The following code creates such a spike train with some arbitrary spike times:
    
    import numpy as np

    spike_train = np.array([0.1, 0.3, 0.45, 0.6, 0.9])

### Loading from text files

Typically, spike train data is loaded into PySpike from data files.
The most straight-forward data files are text files where each line represents one spike train given as an sequence of spike times.
An exemplary file with several spike trains is [PySpike_testdata.txt](https://github.com/mariomulansky/PySpike/blob/master/examples/PySpike_testdata.txt).
To quickly obtain spike trains from such files, PySpike provides the function `load_spike_trains_from_txt`.

    import numpy as np
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 
                                                  time_interval=(0,4000))

This function expects the name of the data file as first parameter, and additionally the time intervall of the spike train measurement can be provided as a pair of start- and end-time values.
If the time interval is provided (`time_interval is not None`), auxiliary spikes at the start- and end-time of the interval are added to the spike trains.
Furthermore, the spike trains are ordered via `np.sort` (disable this feature by providing `sort=False` as a parameter to the load function).
As result, `load_spike_trains_from_txt` returns a *list of arrays* containing the spike trains in the text file.

If you load spike trains yourself, i.e. from data files with different structure, you can use the helper function `add_auxiliary_spikes` to add the auxiliary spikes at the beginning and end of the observation interval.
Both the ISI and the SPIKE distance computation require the presence of auxiliary spikes, so make sure you have those in your spike trains:

    spike_train = spk.add_auxiliary_spikes(spike_train, (T_start, T_end))
    # you provide only a single value, it is interpreted as T_end, while T_start=0
    spike_train = spk.add_auxiliary_spikes(spike_train, T_end)

## Computing bi-variate distances

----------------------
**Important note:**

>Spike trains are expected to be *ordered sequences*! 
>For performance reasons, the PySpike distance functions do not check if the spike trains provided are indeed ordered.
>Make sure that all your spike trains are ordered.
>If in doubt, use `spike_train = np.sort(spike_train)` to obtain a correctly ordered spike train.
>
>Furthermore, the spike trains should have auxiliary spikes at the beginning and end of the observation interval.
>You can ensure this by providing the `time_interval` in the `load_spike_trains_from_txt` function, or calling `add_auxiliary_spikes` for your spike trains.
>The spike trains must have *the same* observation interval!

----------------------

### ISI-distance

The following code loads some exemplary spike trains, computes the dissimilarity profile of the ISI-distance of the first two spike trains, and plots it with matplotlib:

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    isi_profile = spk.isi_distance(spike_trains[0], spike_trains[1])
    x, y = isi_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("ISI distance: %.8f" % isi_profil.avrg())
    plt.show()

The ISI-profile is a piece-wise constant function, there the function `isi_distance` returns an instance of the `PieceWiseConstFunc` class.
As above, this class allows you to obtain arrays that can be used to plot the function with `plt.plt`, but also to compute the absolute average, which amounts to the final scalar ISI-distance.

Furthermore, `PieceWiseConstFunc` provides an `add` function that can be used to add piece-wise constant function, and a `mul_scalar` function that rescales the function by a scalar.
This can be used to obtain an average profile:

    isi_profile1.add(isi_profile2)
    isi_profile1.mul_scalar(0.5)
    x, y = isi_profile1.get_plottable_data()
    plt.plot(x, y, label="Average ISI profile")

## Computing multi-variate distances


## Plotting


## Averaging
