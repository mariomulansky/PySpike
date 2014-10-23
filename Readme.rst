PySpike
=======

.. image:: https://travis-ci.org/mariomulansky/PySpike.svg?branch=master
    :target: https://travis-ci.org/mariomulansky/PySpike

PySpike is a Python library for the numerical analysis of spike train similarity. 
Its core functionality is the implementation of the bivariate ISI_ [#]_ and SPIKE_ [#]_. 
Additionally, it provides functions to compute multi-variate SPIKE and ISI distances, as well as averaging and general spike train processing.
All computation intensive parts are implemented in C via cython_ to reach a competitive performance (factor 100-200 over plain Python).

All source codes are published under the BSD_License_.

.. [#] Kreuz T, Haas JS, Morelli A, Abarbanel HDI, Politi A, *Measuring spike train synchrony.* J Neurosci Methods 165, 151 (2007)

.. [#] Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F, *Monitoring spike train synchrony.* J Neurophysiol 109, 1457 (2013)

Requirements and Installation
-----------------------------

To use PySpike you need Python installed with the following additional packages:

- numpy
- scipy
- matplotlib
- cython
- nosetests (for running the tests)

In particular, make sure that cython_ is configured properly and able to locate a C compiler, otherwise you will only be able to use the much slower plain Python implementations.

To install PySpike, simply download the source, e.g. from Github, and run the :code:`setup.py` script:

.. code:: bash

    git clone https://github.com/mariomulansky/PySpike.git
    cd PySpike
    python setup.py build_ext --inplace

Then you can run the tests using the `nosetests` test framework:

.. code:: bash

    nosetests

Finally, you should make PySpike's installation folder known to Python to be able to import pyspike in your own projects.
Therefore, add your :code:`/path/to/PySpike` to the :code:`$PYTHONPATH` environment variable.

Spike trains
------------

In PySpike, spike trains are represented by one-dimensional numpy arrays containing the sequence of spike times as double values.
The following code creates such a spike train with some arbitrary spike times:
    
.. code:: python

    import numpy as np

    spike_train = np.array([0.1, 0.3, 0.45, 0.6, 0.9])

Loading from text files
.......................

Typically, spike train data is loaded into PySpike from data files.
The most straight-forward data files are text files where each line represents one spike train given as an sequence of spike times.
An exemplary file with several spike trains is `PySpike_testdata.txt <https://github.com/mariomulansky/PySpike/blob/master/examples/PySpike_testdata.txt>`_.
To quickly obtain spike trains from such files, PySpike provides the function :code:`load_spike_trains_from_txt`.

.. code:: python

    import numpy as np
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 
                                                  time_interval=(0, 4000))

This function expects the name of the data file as first parameter, and additionally the time intervall of the spike train measurement can be provided as a pair of start- and end-time values.
If the time interval is provided (:code:`time_interval is not None`), auxiliary spikes at the start- and end-time of the interval are added to the spike trains.
Furthermore, the spike trains are ordered via :code:`np.sort` (disable this feature by providing :code:`sort=False` as a parameter to the load function).
As result, :code:`load_spike_trains_from_txt` returns a *list of arrays* containing the spike trains in the text file.

If you load spike trains yourself, i.e. from data files with different structure, you can use the helper function :code:`add_auxiliary_spikes` to add the auxiliary spikes at the beginning and end of the observation interval.
Both the ISI and the SPIKE distance computation require the presence of auxiliary spikes, so make sure you have those in your spike trains:

.. code:: python

    spike_train = spk.add_auxiliary_spikes(spike_train, (T_start, T_end))
    # if you provide only a single value, it is interpreted as T_end, while T_start=0
    spike_train = spk.add_auxiliary_spikes(spike_train, T_end)

Computing bi-variate distances profiles
---------------------------------------

**Important note:**

------------------------------

    Spike trains are expected to be *ordered sequences*! 
    For performance reasons, the PySpike distance functions do not check if the spike trains provided are indeed ordered.
    Make sure that all your spike trains are ordered.
    If in doubt, use :code:`spike_train = np.sort(spike_train)` to obtain a correctly ordered spike train.
    
    Furthermore, the spike trains should have auxiliary spikes at the beginning and end of the observation interval.
    You can ensure this by providing the :code:`time_interval` in the :code:`load_spike_trains_from_txt` function, or calling :code:`add_auxiliary_spikes` for your spike trains.
    The spike trains must have *the same* observation interval!

----------------------

ISI-distance
............

The following code loads some exemplary spike trains, computes the dissimilarity profile of the ISI-distance of the first two spike trains, and plots it with matplotlib:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    isi_profile = spk.isi_profile(spike_trains[0], spike_trains[1])
    x, y = isi_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("ISI distance: %.8f" % isi_profil.avrg())
    plt.show()

The ISI-profile is a piece-wise constant function, there the function :code:`isi_profile` returns an instance of the :code:`PieceWiseConstFunc` class.
As shown above, this class allows you to obtain arrays that can be used to plot the function with :code:`plt.plt`, but also to compute the time average, which amounts to the final scalar ISI-distance.
By default, the time average is computed for the whole :code:`PieceWiseConstFunc` function.
However, it is also possible to obtain the average of some interval by providing a pair of floats defining the start and end of the interval.
In the above example, the following code computes the ISI-distances obtained from averaging the ISI-profile over four different intervals:

.. code:: python

    isi1 = isi_profil.avrg(interval=(0,1000))
    isi2 = isi_profil.avrg(interval=(1000,2000))
    isi3 = isi_profil.avrg(interval=(2000,3000))
    isi4 = isi_profil.avrg(interval=(3000,4000))

If you are only interested in the scalar ISI-distance and not the profile, you can simly use:

.. code:: python

     isi_dist = spk.isi_distance(spike_trains[0], spike_trains[1], interval)

where :code:`interval` is optional, as above, and if omitted the ISI-distance is computed for the complete spike trains.

Furthermore, PySpike provides the :code:`average_profile` function that can be used to compute the average profile of a list of given :code:`PieceWiseConstFunc` instances.

.. code:: python

    isi_profile1 = spk.isi_profile(spike_trains[0], spike_trains[1])
    isi_profile2 = spk.isi_profile(spike_trains[0], spike_trains[2])
    isi_profile3 = spk.isi_profile(spike_trains[1], spike_trains[2])

    avrg_profile = spk.average_profile([isi_profile1, isi_profile2, isi_profile3])
    x, y = avrg_profile.get_plottable_data()
    plt.plot(x, y, label="Average ISI profile")

Note the difference between the :code:`average_profile` function, which returns a :code:`PieceWiseConstFunc` (or :code:`PieceWiseLinFunc`, see below), and the :code:`avrg` member function above, that computes the integral over the time profile resulting in a single value.
So to obtain overall average ISI-distance of a list of ISI profiles you can first compute the average profile using :code:`average_profile` and the use 

.. code:: python

    avrg_isi = avrg_profile.avrg()

to obtain the final, scalar average ISI distance of the whole set (see also "Computing multi-variate distance" below).


SPIKE-distance
..............

To computation for the spike distance you use the function :code:`spike_profile` instead of :code:`isi_profile` above. 
But the general approach is very similar:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    spike_profile = spk.spike_profile(spike_trains[0], spike_trains[1])
    x, y = spike_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("SPIKE distance: %.8f" % spike_profil.avrg())
    plt.show()

This short example computes and plots the SPIKE-profile of the first two spike trains in the file :code:`PySpike_testdata.txt`.
In contrast to the ISI-profile, a SPIKE-profile is a piece-wise *linear* function and thusly represented by a :code:`PieceWiseLinFunc` object.
Just like the :code:`PieceWiseconstFunc` for the ISI-profile, the :code:`PieceWiseLinFunc` provides a :code:`get_plottable_data` member function that returns array that can be used directly to plot the function.
Furthermore, the :code:`avrg` member function returns the average of the profile defined as the overall SPIKE distance.
As above, you can provide an interval as a pair of floats to :code:`avrg` to specify the averaging interval if required.

Again, you can use

.. code:: python

    spike_dist = spk.spike_distance(spike_trains[0], spike_trains[1], interval)

to compute the SPIKE distance directly, if you are not interested in the profile at all.
:code:`interval` is optional and defines the averaging interval, if neglected the whole spike train is used.
Furthmore, you can use the :code:`average_profile` function to compute an average profile of a list of SPIKE-profiles:

.. code:: python
    
    avrg_profile = spk.average_profile([spike_profile1, spike_profile2, 
                                        spike_profile3])
    x, y = avrg_profile.get_plottable_data()
    plt.plot(x, y, label="Average SPIKE profile")


Computing multi-variate profiles and distances
----------------------------------------------

To compute the multi-variate ISI- or SPIKE-profile of a set of spike trains, you can compute all bi-variate profiles separately and then use the :code:`average_profile` function above.
However, PySpike provides convenience functions for that purpose.
The following example computes the multivariate ISI- and SPIKE-profile for a list of spike trains:

.. code:: python

    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    avrg_isi_profile = spk.isi_profile_multi(spike_trains)
    avrg_spike_profile = spk.spike_profile_multi(spike_trains)

Both functions take an optional parameter :code:`indices`, a list of indices that allows to define the spike trains that should be used for the multi-variate profile.
As before, if you are only interested in the distance values, and not in the profile, PySpike offers the functions: :code:`isi_distance_multi` and :code:`spike_distance_multi`, that return the scalar multi-variate ISI- and SPIKE-distance.
Both distance functions also accept an :code:`interval` parameter that can be used to specify the averaging interval as a pair of floats, if neglected the complete interval is used.

Another option to address large sets of spike trains are distance matrices.
Each entry in the distance matrix represents a bi-variate distance of the spike trains.
Hence, the distance matrix is symmetric and has zero values at the diagonal.
The following example computes and plots the ISI- and SPIKE-distance matrix, where for the latter one only the time interval T=0..1000 is used for the averaging.

.. code:: python

    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt", 4000)

    plt.figure()
    isi_distance = spk.isi_distance_matrix(spike_trains)
    plt.imshow(isi_distance, interpolation='none')
    plt.title("ISI-distance")
    
    plt.figure()
    spike_distance = spk.spike_distance_matrix(spike_trains, interval=(0,1000))
    plt.imshow(spike_distance, interpolation='none')
    plt.title("SPIKE-distance")

    plt.show()


Time Averages
-------------




.. _ISI: http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony#ISI-distance
.. _SPIKE: http://www.scholarpedia.org/article/SPIKE-distance
.. _cython: http://www.cython.org
.. _BSD_License: http://opensource.org/licenses/BSD-2-Clause
