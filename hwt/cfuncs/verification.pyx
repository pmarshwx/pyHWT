cimport cython
import numpy as np
cimport numpy as np
import pdb

cdef extern from 'math.h':
    float sqrt(float x)


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
                                     unsigned int n):
    
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
        rxx = min(xlen -1, i + n)
        rxn = max(0, i - n)
        for j in xrange(ylen):
            ryx = min(ylen - 1, j + n)
            ryn = max(0, j - n)
            # rxx/ryx + 1 to be included in xrange
            for ii in xrange(rxn, rxx + 1):
                for jj in xrange(ryn, ryx + 1):
                    if data[ii, jj] == 1:
                        frac[i, j ] += 1

    frac = frac/(n*n)

    return frac


@cython.boundscheck(False)
@cython.cdivision(True)
cdef get_fss_cfraction(np.ndarray[DTYPE_t, ndim=2] data,
                                       float roi,
                                       float dx):

    cdef unsigned int xlen = data.shape[0]
    cdef unsigned int ylen = data.shape[1]
    cdef unsigned int ng
    cdef int jw, je, isouth, inorth, n
    cdef float rng, distsq, dist
    cdef Py_ssize_t i, j, ii, jj

    cdef np.ndarray[DTYPE64_t, ndim=2] frac = np.zeros([xlen, ylen], dtype=DTYPE64)
    n = 0

    rng = roi/dx
    ng = int(rng)

    for i in range(xlen):
        isouth = i-ng
        inorth = i+ng + 1
        for j in range(ylen):
            jw = j-ng
            je = j+ng + 1
            for ii in range(isouth, inorth):
                for jj in range(jw, je):
                    if jw < 0 or je >= ylen or isouth < 0 or inorth >= xlen:
                        continue
                    distsq = float(j-jj)*float(j-jj)  + float(i-ii)*float(i-ii)
                    dist = sqrt(distsq)
                    if dist <= rng:
                        n += 1
                        if data[ii, jj] == 1:
                            frac[i,j] += 1

    if n == 0:
        raise ValueError('No points in neighborhood. Check radius and grid spacing.')
    else:
        frac = frac/float(n)

    return frac


@cython.boundscheck(False)
@cython.cdivision(True)
cdef get_fss_mse(np.ndarray[DTYPE64_t, ndim = 2] obs, 
                               np.ndarray[DTYPE64_t, ndim = 2] fcst):

    cdef Py_ssize_t i, j
    cdef unsigned int nx, ny
    cdef float mse = 0

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

    ref = ref/(nx*ny)

    return ref



def fss(obs, fcst, r = None, dx = None, neighborhood = None):
    """
    Calculate Fractions Skill Score.

    Parameters
    ----------
    obs : Observation binary array
    fcst : Forecast binary array
    r : For a square neighborhood, radius in grid points. For
        a circle neighborhood, radius of influence in kilometers.
    dx : Grid spacing. This is only relevant for circular
        neighborhoods.
    neighborhood : Shape of neighborhood to calculate
        fraction of hits. Options are 'square' or 'circle'. Default
        is None which simply does a grid point to grid point 
        comparison.

    Returns
    -------
    fss : scalar

    Raises
    ------
    ValueError
        The neighborhood size is negative, the input arrays are
        difference sizes, or the neighborhood type is unknown.

    Notes
    -----
    - Square neighborhoods do not use any distance calculations. They
        simply have r grid points beyond each point in their neighborhood.
    - Circle neighborhoods do use distance (via dx) to calculate a true
        radius of influence.
    """

    if neighborhood is None:
        obs = obs.astype(np.float64)
        fcst = fcst.astype(np.float64)
        mse = get_fss_mse(obs, fcst)
        ref = get_fss_ref(obs, fcst)
    elif neighborhood == 'square':
        if r <= 0:
            raise ValueError('Neighborhood radius must be non-zero and positive.')
        else:
            ofrac = get_fss_fraction(obs, r)
            ffrac = get_fss_fraction(fcst, r)
            mse = get_fss_mse(ofrac, ffrac)
            ref = get_fss_ref(ofrac, ffrac)
    elif neighborhood == 'circle':
        if dx is None or r is None:
            raise ValueError('Missing grid spacing or radius.')
        else:
            if dx <= 0 or r <= 0:
                raise ValueError('Grid spacing and radius must be non-zero and positive.')
            else:
                ofrac = get_fss_cfraction(obs, r, dx)
                ffrac = get_fss_cfraction(fcst, r, dx)
                mse = get_fss_mse(ofrac, ffrac)
                ref = get_fss_ref(ofrac, ffrac)
    else:
        raise ValueError('Invalid neighborhood type.')

    fss = 1 - mse/ref

    return fss