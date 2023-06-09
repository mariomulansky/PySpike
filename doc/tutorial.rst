Spike trains
------------

In PySpike, spike trains are represented by :class:`.SpikeTrain` objects.
A :class:`.SpikeTrain` object consists of the spike times given as `numpy` arrays as well as the edges of the spike train as :code:`[t_start, t_end]`.
The following code creates such a spike train with some arbitrary spike times:
    
.. code:: python

    import numpy as np
    from pyspike import SpikeTrain

    spike_train = SpikeTrain(np.array([0.1, 0.3, 0.45, 0.6, 0.9], [0.0, 1.0]))

Loading from text files
.......................

Typically, spike train data is loaded into PySpike from data files.
The most straight-forward data files are text files where each line represents one spike train given as an sequence of spike times.
An exemplary file with several spike trains is `PySpike_testdata.txt <https://github.com/mariomulansky/PySpike/blob/master/examples/PySpike_testdata.txt>`_.
To quickly obtain spike trains from such files, PySpike provides the function :func:`.load_spike_trains_from_txt`.

.. code:: python

    import numpy as np
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 
                                                  edges=(0, 4000))

This function expects the name of the data file as first parameter.
Furthermore, the time interval of the spike train measurement (edges of the spike trains) should be provided as a pair of start- and end-time values.
Furthermore, the spike trains are sorted via :code:`np.sort` (disable this feature by providing :code:`is_sorted=True` as a parameter to the load function).
As result, :func:`.load_spike_trains_from_txt` returns a *list of arrays* containing the spike trains in the text file.


**Important note:**

------------------------------

Spike trains are expected to be *sorted*! 
For performance reasons, the PySpike distance functions do not check if the spike trains provided are indeed sorted.
Make sure that all your spike trains are sorted, which is ensured if you use the :func:`.load_spike_trains_from_txt` function with the parameter `is_sorted=False` (default).
If in doubt, use :meth:`.SpikeTrain.sort()` to ensure a correctly sorted spike train.

Alternatively the function :func:`.reconcile_spike_trains` applies three fixes to a list of SpikeTrain objects. It sorts
the times, it removes all but one of any duplicated time, and it ensures all t_start and t_end values are compatible

.. code:: python

    from pyspike.spikes import reconcile_spike_trains
    spike_trains = reconcile_spike_trains(spike_trains)

If you need to copy a spike train, use the :meth:`.SpikeTrain.copy()` method.
Simple assignment `t2 = t1` does not create a copy of the spike train data, but a reference as `numpy.array` is used for storing the data.

PySpike algorithms 
-------------------

PySpike supports four basic algorithms for comparing spike trains and their adaptive generalizations

The basic algorithms are:

1) ISI-distance  (Inter Spike Intervals)
2) SPIKE-distance
3) Rate-Independent SPIKE-distance (RI-SPIKE)
4) SPIKE sychronization

plus

(5-8) Adaptive generalizations of 1-4 based on the MRTS (Minimum Relevant Time Scale) parameter

Algorithms 3 and 5-8 are new in version 0.8.0.

ISI-distance
............

The following code loads some exemplary spike trains, computes the dissimilarity profile of the ISI-distance of the first two :class:`.SpikeTrain` s, and plots it with matplotlib:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  edges=(0, 4000))
    isi_profile = spk.isi_profile(spike_trains[0], spike_trains[1])
    x, y = isi_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("ISI distance: %.8f" % isi_profile.avrg())
    plt.show()

The ISI-profile is a piece-wise constant function, and hence the function :func:`.isi_profile` returns an instance of the :class:`.PieceWiseConstFunc` class.
As shown above, this class allows you to obtain arrays that can be used to plot the function with :code:`plt.plt`, but also to compute the time average, which amounts to the final scalar ISI-distance.
By default, the time average is computed for the whole :class:`.PieceWiseConstFunc` function.
However, it is also possible to obtain the average of a specific interval by providing a pair of floats defining the start and end of the interval.
For the above example, the following code computes the ISI-distances obtained from averaging the ISI-profile over four different intervals:

.. code:: python

    isi1 = isi_profile.avrg(interval=(0, 1000))
    isi2 = isi_profile.avrg(interval=(1000, 2000))
    isi3 = isi_profile.avrg(interval=[(0, 1000), (2000, 3000)])
    isi4 = isi_profile.avrg(interval=[(1000, 2000), (3000, 4000)])

Note, how also multiple intervals can be supplied by giving a list of tuples.

If you are only interested in the scalar ISI-distance and not the profile, you can simply use:

.. code:: python

     isi_dist = spk.isi_distance(spike_trains[0], spike_trains[1], interval=(0, 1000))

where :code:`interval` is optional, as above, and if omitted the ISI-distance is computed for the complete spike train.

SPIKE-distance
..............

To compute for the spike distance profile you use the function :func:`.spike_profile` instead of :code:`isi_profile` above. 
But the general approach is very similar:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  edges=(0, 4000))
    spike_profile = spk.spike_profile(spike_trains[0], spike_trains[1])
    x, y = spike_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("SPIKE distance: %.8f" % spike_profile.avrg())
    plt.show()

This short example computes and plots the SPIKE-profile of the first two spike trains in the file :code:`PySpike_testdata.txt`.

In contrast to the ISI-profile, a SPIKE-profile is a piece-wise *linear* function and is therefore represented by a :class:`.PieceWiseLinFunc` object.
Just like the :class:`.PieceWiseConstFunc` for the ISI-profile, the :class:`.PieceWiseLinFunc` provides a :meth:`.PieceWiseLinFunc.get_plottable_data` member function that returns arrays that can be used directly to plot the function.
Furthermore, the :meth:`.PieceWiseLinFunc.avrg` member function returns the average of the profile defined as the overall SPIKE distance.
As above, you can provide an interval as a pair of floats as well as a sequence of such pairs to :code:`avrg` to specify the averaging interval if required.

Again, you can use:

.. code:: python

    spike_dist = spk.spike_distance(spike_trains[0], spike_trains[1], interval=ival)

to compute the SPIKE distance directly, if you are not interested in the profile at all.
The parameter :code:`interval` is optional and if neglected the whole time interval is used.


Rate-Independent SPIKE-distance
...............................

This variant of the SPIKE-distance disregards any differences in base rates and focuses purely on spike timing.
It can be calculated by setting the optional parameter "RI=True":

.. code:: python

    ri_spike_dist = spk.spike_distance(spike_trains[0], spike_trains[1], RI=True)


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
Therefore, it is represented by a :class:`.DiscreteFunction`.

To compute for the spike synchronization profile, PySpike provides the function :func:`.spike_sync_profile`.
The general handling of the profile, however, is similar to the other profiles above:

.. code:: python

    import matplotlib.pyplot as plt
    import pyspike as spk
    
    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  edges=(0, 4000))
    spike_profile = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
    x, y = spike_profile.get_plottable_data()

For the direct computation of the overall spike synchronization value within some interval, the :func:`.spike_sync` function can be used:

.. code:: python
   
    spike_sync = spk.spike_sync(spike_trains[0], spike_trains[1], interval=ival)


Adaptive generalizations
........................

The adaptive generalizations for all four of these basic measures can be calculated by setting the optional parameter "MRTS=<value>" (MRTS - Minimum Relevant Time Scale).
If <value> is greater than zero the respective basic algorithm is modified to reduce emphasis on smaller spike time differences.
If MRTS is set to 'auto', the threshold is automatically extracted from the data.

Here are a few example lines:

.. code:: python

    a_isi_dist = spk.isi_distance(spike_trains, MRTS=10)

    a_spike_profile = spk.spike_profile(spike_trains, MRTS=20)

    a_ri_spike_matrix = spk.spike_distance_matrix(spike_trains[0], spike_trains[1], RI=True, MRTS=50)

    a_spike_sync_auto = spk.spike_sync(spike_trains[0], spike_trains[1], MRTS='auto')


Computing multivariate profiles and distances
----------------------------------------------

To compute the multivariate ISI-profile, SPIKE-profile or SPIKE-Synchronization profile for a set of spike trains, simply provide a list of spike trains to the profile or distance functions.
The following example computes the multivariate ISI-, SPIKE- and SPIKE-Sync-profile for a list of spike trains:

.. code:: python

    spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                                  edges=(0, 4000))
    avrg_isi_profile = spk.isi_profile(spike_trains)
    avrg_spike_profile = spk.spike_profile(spike_trains)
    avrg_spike_sync_profile = spk.spike_sync_profile(spike_trains)

All functions also take an optional parameter :code:`indices`, a list of indices that allows to define the spike trains that should be used for the multivariate profile.
As before, if you are only interested in the distance values, and not in the profile, you can call the functions: :func:`.isi_distance`, :func:`.spike_distance` and :func:`.spike_sync` with a list of spike trains.
They return the scalar overall multivariate ISI-, SPIKE-distance or the SPIKE-Synchronization value.

The following code is equivalent to the bivariate example above, computing the ISI-Distance between the first two spike trains in the given interval using the :code:`indices` parameter:

.. code:: python

     isi_dist = spk.isi_distance(spike_trains, indices=[0, 1], interval=(0, 1000))

As you can see, the distance functions also accept an :code:`interval` parameter that can be used to specify the begin and end of the averaging interval as a pair of floats, if neglected the complete interval is used.

**Note:**

------------------------------

    Instead of providing lists of spike trains to the profile or distance functions, you can also call those functions with many spike trains as (unnamed) parameters, e.g.:
    
    .. code:: python
       
       # st1, st2, st3, st4 are spike trains
       spike_prof = spk.spike_profile(st1, st2, st3, st4)
    
------------------------------


Another option to characterize large sets of spike trains are distance matrices.
Each entry in the distance matrix represents a bivariate distance (similarity for SPIKE-Synchronization) of two spike trains.
The distance matrix is symmetric and has zero values (ones) at the diagonal and is computed with the functions :func:`.isi_distance_matrix`, :func:`.spike_distance_matrix` and :func:`.spike_sync_matrix`.
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


Quantifying Leaders and Followers: Spike Train Order
-----------------------------------------------------

PySpike provides functionality to quantify how much a set of spike trains
resembles a synfire pattern (ie perfect leader-follower pattern). For details
on the algorithms please see
`our article in NJP <http://iopscience.iop.org/article/10.1088/1367-2630/aa68c3>`_.

The following example computes the Spike Order profile and Synfire Indicator
of two Poissonian spike trains.

.. code:: python

    import numpy as np
    from matplotlib import pyplot as plt
    import pyspike as spk


    st1 = spk.generate_poisson_spikes(1.0, [0, 20])
    st2 = spk.generate_poisson_spikes(1.0, [0, 20])

    d = spk.spike_directionality(st1, st2)

    print "Spike Directionality of two Poissonian spike trains:", d

    E = spk.spike_train_order_profile(st1, st2)

    plt.figure()
    x, y = E.get_plottable_data()
    plt.plot(x, y, '-ob')
    plt.ylim(-1.1, 1.1)
    plt.xlabel("t")
    plt.ylabel("E")
    plt.title("Spike Train Order Profile")

    plt.show()

Additionally, PySpike can also compute the optimal ordering of the spike trains,
ie the ordering that most resembles a synfire pattern. The following example
computes the optimal order of a set of 20 Poissonian spike trains:

.. code:: python

    M = 20
    spike_trains = [spk.generate_poisson_spikes(1.0, [0, 100]) for m in xrange(M)]

    F_init = spk.spike_train_order(spike_trains)
    print "Initial Synfire Indicator for 20 Poissonian spike trains:", F_init

    D_init = spk.spike_directionality_matrix(spike_trains)
    phi, _ = spk.optimal_spike_train_sorting(spike_trains)
    F_opt = spk.spike_train_order(spike_trains, indices=phi)
    print "Synfire Indicator of optimized spike train sorting:", F_opt

    D_opt = spk.permutate_matrix(D_init, phi)

    plt.figure()
    plt.imshow(D_init)
    plt.title("Initial Directionality Matrix")

    plt.figure()
    plt.imshow(D_opt)
    plt.title("Optimized Directionality Matrix")

    plt.show()
