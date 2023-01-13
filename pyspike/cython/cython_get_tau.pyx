#cython language_level=3
from libc.math cimport fmin

cdef double Interpolate(a, b, t):
    """ thresholded interpolation
        If t small, return min(a,b)
        if t big, return b
        in between, return t      
    """
    if t < a and a < b:  return a
    if t < b and b <= a: return b
    if t > b:            return b    
    return t               # interpolation

cdef double get_tau(double[:] spikes1, double[:] spikes2,
                    int i, int j, double max_tau, double MRTS):
    """ Compute coincidence window
        In: spikes1, spikes2 - times of two spike trains
            i, j - indices into spikes1, spikes2 to compare
            max_tau - maximum size of threshold
            MRTS - adaptation parameter  
        out: combined coincidence window (Eq 19 in reference)
    """

    ## "distances" to neighbor: F/P=future/past, 1/2=N in spikesN.
    cdef double mF1 = max_tau
    cdef double mP1 = max_tau
    cdef double mF2 = max_tau
    cdef double mP2 = max_tau
    
    if i < len(spikes1)-1 and i > -1:
        mF1 = (spikes1[i+1]-spikes1[i])
    if j < len(spikes2)-1 and j > -1:
        mF2 = (spikes2[j+1]-spikes2[j])
    if i > 0:
        mP1 = (spikes1[i]-spikes1[i-1])
    if j > 0:
        mP2 = (spikes2[j]-spikes2[j-1])

    mF1, mF2, mP1, mP2 = mF1/2., mF2/2., mP1/2., mP2/2.
    MRTS /= 4.

    if i<0 or j<0 or spikes1[i] <= spikes2[j]:
        s1F = Interpolate(mP1, mF1, MRTS)
        s2P = Interpolate(mF2, mP2, MRTS)
        return fmin(s1F, s2P)
    else:
        s1P = Interpolate(mF1, mP1, MRTS)
        s2F = Interpolate(mP2, mF2, MRTS)
        return fmin(s1P, s2F)
