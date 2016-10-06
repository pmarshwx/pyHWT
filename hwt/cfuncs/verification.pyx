cimport cython
import numpy as np
cimport numpy as np


DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t




@cython.boundscheck(False)
def reliability(np.ndarray[DTYPE_t, ndim=2] fcst,
                np.ndarray[DTYPE_t, ndim=2] obs,
                np.ndarray[DTYPE_t, ndim=2] mask,
                unsigned int asize,
                int missing = 9999):

    cdef unsigned int ii = fcst.shape[0]
    cdef unsigned int jj = fcst.shape[1]
    cdef Py_ssize_t i, j

    # The reason we add +2 to asize is to account for the values at 0 and
    # 100. The user passes in negative numbers for those points exactly equal
    # to 0. This becomes the first array element. The second array element are
    # all points that are between 0 and first value. Thus we must add an array
    # element to hold the points actually equal to 0.  Same is true for 100.
    # asize only accounts for points up to the value immediately preceding the
    # last value so we must add another element to have a position for the last
    # element. Thus, we have to add 2 additional elements to account for
    # everything
    cdef np.ndarray[DTYPE2_t, ndim=1] fhist = np.zeros(asize+2, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=1] ohist = np.zeros(asize+2, dtype=DTYPE2)

    for i in range(ii):
        for j in range(jj):
            if (mask[i,j] == 0 or mask[i,j] == missing or fcst[i,j] == missing
                or obs[i,j] == missing):
                    continue
            elif fcst[i,j] < 0:
                fhist[0] += 1
                ohist[0] += obs[i,j]
            else:
                # We must add 1 to the returned value because we added an
                # element to the beginning of the array for all points that
                # were exactlyequal to 0. Elements at this point that are
                # equal to 0 are actually points between 0 and first value.
                fhist[fcst[i,j]+1] += 1
                ohist[fcst[i,j]+1] += obs[i,j]


    return (fhist, ohist)



@cython.boundscheck(False)
@cython.cdivision(True)
def get_contingency(np.ndarray[DTYPE64_t, ndim=2] fcst,
                    np.ndarray[DTYPE64_t, ndim=2] obs,
                    np.ndarray[DTYPE_t, ndim=2] mask,
                    int missing = 9999):

    cdef unsigned int ii = fcst.shape[0]
    cdef unsigned int jj = fcst.shape[1]
    cdef unsigned int a, b, c, d
    cdef Py_ssize_t i, j

    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(ii):
        for j in range(jj):
            if (mask[i,j] == 0 or mask[i,j] == missing or fcst[i,j] == missing
                or obs[i,j] == missing):
                    continue
            elif fcst[i,j] == 1:
                if obs[i,j] == 1:
                    a += 1
                else:
                    b += 1
            else:
                if obs[i,j] == 1:
                    c += 1
                else:
                    d += 1

    return (a, b, c, d)



@cython.boundscheck(False)
@cython.cdivision(True)
cdef get_fss_fraction(np.ndarray[DTYPE_t, ndim=2] data, 
                                     unsigned int roi):
    
    cdef Py_ssize_t i, j, ii, jj
    cdef unsigned int xlen = data.shape[0]
    cdef unsigned int ylen = data.shape[1]
    cdef unsigned int rxx
    cdef unsigned int rxn
    cdef unsigned int ryx
    cdef unsigned int ryn
    cdef np.ndarray[DTYPE64_t, ndim=2] frac = np.zeros([xlen, ylen], dtype=DTYPE64)

    for i in xrange(xlen):
        # xlen/ylen - 1 to give proper index value
        # for min functions
        rxx = min(xlen -1, i + roi)
        rxn = max(0, i - roi)
        for j in xrange(ylen):
            ryx = min(ylen - 1, j + roi)
            ryn = max(0, j - roi)
            # rxx/ryx + 1 to be included in xrange
            for ii in xrange(rxn, rxx + 1):
                for jj in xrange(ryn, ryx + 1):
                    if data[ii, jj] == 1:
                        frac[i, j ] += 1

    frac = frac/roi**2

    return frac



@cython.boundscheck(False)
@cython.cdivision(True)
cdef get_fss_mse(np.ndarray[DTYPE64_t, ndim = 2] obs, 
                               np.ndarray[DTYPE64_t, ndim = 2] fcst):

    cdef Py_ssize_t i, j
    cdef unsigned int nx, ny
    cdef double mse = 0

    if (obs.shape[0] != fcst.shape[0]) or (obs.shape[1] != fcst.shape[1]):
        raise ValueError('Observation and forecast arrays must be the same shape.')

    nx = obs.shape[0]
    ny = obs.shape[1]

    for i in xrange(nx):
        for j in xrange(ny):
            mse = mse + (obs[i,j] - fcst[i,j])*(obs[i,j] - fcst[i,j])

    mse = mse/(nx*ny)

    return mse



@cython.boundscheck(False)
@cython.cdivision(True)
cdef get_fss_ref(np.ndarray[DTYPE64_t, ndim = 2] obs, 
                             np.ndarray[DTYPE64_t, ndim = 2] fcst):

    cdef Py_ssize_t i, j
    cdef unsigned int nx, ny
    cdef double ref = 0

    if (obs.shape[0] != fcst.shape[0]) or (obs.shape[1] != fcst.shape[1]):
        raise ValueError('Observation and forecast arrays must be the same shape.')

    nx = obs.shape[0]
    ny = obs.shape[1]

    for i in xrange(nx):
        for j in xrange(ny):
            ref = ref + obs[i,j]*obs[i,j] + fcst[i,j]*fcst[i,j]

    ref = ref/(nx/ny)

    return ref



def fss(obs, fcst, n = 0):

    if n < 0:
        raise ValueError('Neighborhood radius cannot be negative.')

    if n > 0:
        ofrac = get_fss_fraction(obs, n)
        ffrac = get_fss_fraction(fcst, n)

        mse = get_fss_mse(ofrac, ffrac)
        ref = get_fss_ref(ofrac, ffrac)
    else:
        mse = get_fss_mse(obs, fcst)
        ref = get_fss_ref(obs, fcst)

    fss = 1 - mse/ref

    return fss