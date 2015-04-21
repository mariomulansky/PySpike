PySpike
=======

.. image:: https://travis-ci.org/mariomulansky/PySpike.svg?branch=master
    :target: https://travis-ci.org/mariomulansky/PySpike

PySpike is a Python library for the numerical analysis of spike train similarity. 
Its core functionality is the implementation of the bivariate ISI_ and SPIKE_ distance [#]_ [#]_ as well as SPIKE-Synchronization_ [#]_.
Additionally, it provides functions to compute multivariate profiles, distance matrices, as well as averaging and general spike train processing.
All computation intensive parts are implemented in C via cython_ to reach a competitive performance (factor 100-200 over plain Python).

PySpike provides the same fundamental functionality as the SPIKY_ framework for Matlab, which additionally contains spike-train generators, more spike train distance measures and many visualization routines.

All source codes are available on `Github <https://github.com/mariomulansky/PySpike>`_  and are published under the BSD_License_.

.. [#] Kreuz T, Haas JS, Morelli A, Abarbanel HDI, Politi A, *Measuring spike train synchrony.* J Neurosci Methods 165, 151 (2007) `[pdf] <http://wwwold.fi.isc.cnr.it/users/thomas.kreuz/images/Kreuz_JNeurosciMethods_2007_Spike-Train-Synchrony.pdf>`_

.. [#] Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F, *Monitoring spike train synchrony.* J Neurophysiol 109, 1457 (2013) `[pdf] <http://wwwold.fi.isc.cnr.it/users/thomas.kreuz/images/Kreuz_JNeurophysiol_2013_SPIKE-distance.pdf>`_

.. [#] Kreuz T, Mulansky M and Bozanic N, *SPIKY: A graphical user interface for monitoring spike train synchrony*, tbp (2015)

Requirements and Installation
-----------------------------

PySpike is available at Python Package Index and this is the easiest way to obtain the PySpike package.
If you have `pip` installed, just run

.. code:: bash

   sudo pip install pyspike

to install pyspike.
PySpike requires `numpy` as minimal requirement, as well as a C compiler to generate the binaries.

Install from Github sources
...........................

You can also obtain the latest PySpike developer version from the github repository.
For that, make sure you have the following Python libraries installed:

- numpy
- cython
- matplotlib (for the examples)
- nosetests (for running the tests)

In particular, make sure that cython_ is configured properly and able to locate a C compiler, otherwise PySpike will use the much slower Python implementations.

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

This function expects the name of the data file as first parameter.
Additionally, the time interval of the spike train measurement can be provided as a pair of start- and end-time values.
If the time interval is provided (:code:`time_interval is not None`), auxiliary spikes at the start- and end-time of the interval are added to the spike trains.
Furthermore, the spike trains are sorted via :code:`np.sort` (disable this feature by providing :code:`is_sorted=True` as a parameter to the load function).
As result, :code:`load_spike_trains_from_txt` returns a *list of arrays* containing the spike trains in the text file.

If you load spike trains yourself, i.e. from data files with different structure, you can use the helper function :code:`add_auxiliary_spikes` to add the auxiliary spikes at the beginning and end of the observation interval.
Both the ISI and the SPIKE distance computation require the presence of auxiliary spikes, so make sure you have those in your spike trains:

.. code:: python

    spike_train = spk.add_auxiliary_spikes(spike_train, (T_start, T_end))
    # if you provide only a single value, it is interpreted as T_end, while T_start=0
    spike_train = spk.add_auxiliary_spikes(spike_train, T_end)

Computing bivariate distances profiles
---------------------------------------

**Important note:**

------------------------------

    Spike trains are expected to be *sorted*! 
    For performance reasons, the PySpike distance functions do not check if the spike trains provided are indeed sorted.
    Make sure that all your spike trains are sorted, which is ensured if you use the `load_spike_trains_from_txt` function with the parameter `is_sorted=False`.
    If in doubt, use :code:`spike_train = np.sort(spike_train)` to obtain a correctly sorted spike train.
    
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
    print("ISI distance: %.8f" % isi_profile.avrg())
    plt.show()

The ISI-profile is a piece-wise constant function, and hence the function :code:`isi_profile` returns an instance of the :code:`PieceWiseConstFunc` class.
As shown above, this class allows you to obtain arrays that can be used to plot the function with :code:`plt.plt`, but also to compute the time average, which amounts to the final scalar ISI-distance.
By default, the time average is computed for the whole :code:`PieceWiseConstFunc` function.
However, it is also possible to obtain the average of a specific interval by providing a pair of floats defining the start and end of the interval.
In the above example, the following code computes the ISI-distances obtained from averaging the ISI-profile over four different intervals:

.. code:: python

    isi1 = isi_profile.avrg(interval=(0, 1000))
    isi2 = isi_profile.avrg(interval=(1000, 2000))
    isi3 = isi_profile.avrg(interval=[(0, 1000), (2000, 3000)])
    isi4 = isi_profile.avrg(interval=[(1000, 2000), (3000, 4000)])

Note, how also multiple intervals can be supplied by giving a list of tuples.

If you are only interested in the scalar ISI-distance and not the profile, you can simply use:

.. code:: python

     isi_dist = spk.isi_distance(spike_trains[0], spike_trains[1], interval)

where :code:`interval` is optional, as above, and if omitted the ISI-distance is computed for the complete spike trains.


SPIKE-distance
..............

To compute for the spike distance profile you use the function :code:`spike_profile` instead of :code:`isi_profile` above. 
But the general approach is very similar:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    spike_profile = spk.spike_profile(spike_trains[0], spike_trains[1])
    x, y = spike_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("SPIKE distance: %.8f" % spike_profile.avrg())
    plt.show()

This short example computes and plots the SPIKE-profile of the first two spike trains in the file :code:`PySpike_testdata.txt`.
In contrast to the ISI-profile, a SPIKE-profile is a piece-wise *linear* function and is therefore represented by a :code:`PieceWiseLinFunc` object.
Just like the :code:`PieceWiseConstFunc` for the ISI-profile, the :code:`PieceWiseLinFunc` provides a :code:`get_plottable_data` member function that returns arrays that can be used directly to plot the function.
Furthermore, the :code:`avrg` member function returns the average of the profile defined as the overall SPIKE distance.
As above, you can provide an interval as a pair of floats as well as a sequence of such pairs to :code:`avrg` to specify the averaging interval if required.

Again, you can use

.. code:: python

    spike_dist = spk.spike_distance(spike_trains[0], spike_trains[1], interval)

to compute the SPIKE distance directly, if you are not interested in the profile at all.
The parameter :code:`interval` is optional and if neglected the whole spike train is used.


SPIKE synchronization
.....................

**Important note:**

------------------------------

    SPIKE-Synchronization measures *similarity*. 
    That means, a value of zero indicates absence of synchrony, while a value of one denotes the presence of synchrony.
    This is exactly opposite to the other two measures: ISI- and SPIKE-distance.

----------------------


SPIKE synchronization is another approach to measure spike synchrony.
In contrast to the SPIKE- and ISI-distance, it measures similarity instead of dissimilarity, i.e. higher values represent larger synchrony.
Another difference is that the SPIKE synchronization profile is only defined exactly at the spike times, not for the whole interval of the spike trains.
Therefore, it is represented by a :code:`DiscreteFunction`.

To compute for the spike synchronization profile, PySpike provides the function :code:`spike_sync_profile`.
The general handling of the profile, however, is similar to the other profiles above:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    spike_profile = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
    x, y = spike_profile.get_plottable_data()

For the direct computation of the overall spike synchronization value within some interval, the :code:`spike_sync` function can be used:

.. code:: python
   
   spike_sync = spk.spike_sync(spike_trains[0], spike_trains[1], interval)


Computing multivariate profiles and distances
----------------------------------------------

To compute the multivariate ISI-profile, SPIKE-profile or SPIKE-Synchronization profile f a set of spike trains, PySpike provides multi-variate version of the profile function.
The following example computes the multivariate ISI-, SPIKE- and SPIKE-Sync-profile for a list of spike trains:

.. code:: python

    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    avrg_isi_profile = spk.isi_profile_multi(spike_trains)
    avrg_spike_profile = spk.spike_profile_multi(spike_trains)
    avrg_spike_sync_profile = spk.spike_sync_profile_multi(spike_trains)

All functions take an optional parameter :code:`indices`, a list of indices that allows to define the spike trains that should be used for the multivariate profile.
As before, if you are only interested in the distance values, and not in the profile, PySpike offers the functions: :code:`isi_distance_multi`, :code:`spike_distance_multi` and :code:`spike_sync_multi`, that return the scalar overall multivariate ISI- and SPIKE-distance as well as the SPIKE-Synchronization value.
Those functions also accept an :code:`interval` parameter that can be used to specify the begin and end of the averaging interval as a pair of floats, if neglected the complete interval is used.

Another option to characterize large sets of spike trains are distance matrices.
Each entry in the distance matrix represents a bivariate distance (similarity for SPIKE-Synchronization) of two spike trains.
The distance matrix is symmetric and has zero values (ones) at the diagonal.
The following example computes and plots the ISI- and SPIKE-distance matrix as well as the SPIKE-Synchronization-matrix, with different intervals.

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

    plt.figure()
    spike_sync = spk.spike_sync_matrix(spike_trains, interval=(2000,4000))
    plt.imshow(spike_sync, interpolation='none')
    plt.title("SPIKE-Sync")

    plt.show()


===============================================================================

*The work on PySpike was supported by the European Comission through the Marie
Curie Initial Training Network* `Neural Engineering Transformative Technologies
(NETT) <http://www.neural-engineering.eu/>`_ *under the project number 289146.*


**Python/C Programming:**
 - Mario Mulansky

**Scientific Methods:**
 - Thomas Kreuz
 - Nebojsa D. Bozanic
 - Mario Mulansky
 - Conor Houghton
 - Daniel Chicharro

.. _ISI: http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony#ISI-distance
.. _SPIKE: http://www.scholarpedia.org/article/SPIKE-distance
.. _SPIKE-Synchronization: http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony#SPIKE_synchronization
.. _cython: http://www.cython.org
.. _SPIKY: http://wwwold.fi.isc.cnr.it/users/thomas.kreuz/Source-Code/SPIKY.html
.. _BSD_License: http://opensource.org/licenses/BSD-2-Clause
