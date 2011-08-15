cimport cython
import numpy as np
cimport numpy as np


cdef extern from 'math.h':
    float exp(float x)

DTYPE = np.int
DTYPE2 = np.double
DTYPE32 = np.float32
DTYPE64 = np.float64
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE2_t
ctypedef np.float32_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t

@cython.cdivision(True)
cpdef float cdf(float x, float mu, float beta):
    cdef float z
    z = -(x-mu)/beta
    return exp(-exp(z))

@cython.cdivision(True)
cpdef float linear_interp(float thresh, float vmin, float vmax,
        float pmin, float pmax):
    return 1. - (pmin + (((thresh-vmin) / (vmax-vmin)) * (pmax-pmin)))

@cython.boundscheck(False)
@cython.cdivision(True)
def corrected_ensemble(np.ndarray[DTYPE64_t, ndim=3] members, float thresh):

    cdef unsigned int kk = members.shape[0]
    cdef unsigned int ii = members.shape[1]
    cdef unsigned int jj = members.shape[2]
    cdef unsigned int ind = 0
    cdef float mean, sigma, mu, beta, diffs, PI, EULER, SQRT6
    cdef float prob1, prob2
    cdef np.ndarray[DTYPE64_t, ndim=1] vals = np.zeros([kk], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=1] rh = np.zeros([kk+1], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] probs = np.zeros([ii,jj], dtype=DTYPE64)
    cdef Py_ssize_t i, j, k, v

    PI = 3.1415926535897932384
    EULER = 0.5772156649015328606
    SQRT6 = 6**0.5

    for 0 <= k <= kk:
        rh[k] = 1./(kk+1.)

    for 0 <= i < ii:
        for 0 <= j < jj:
            prob1 = 0.
            prob2 = 0.
            for 0 <= k < kk:
                vals[k] = members[k,i,j]
            vals.sort()

            # Create Ensemble Probabilities Less Than Envelope
            if thresh < vals[0]:
                probs[i,j] = linear_interp(thresh, 0., vals[0], 0., rh[0])

            # Create Ensemble Probabilities In Envelope
            elif thresh < vals[-1]:
                for 0 <= v < kk:
                    if thresh < vals[v]: break
                    ind = v
                if thresh == vals[v]:
                    for 0 <= v <= ind:
                        probs[i,j] += rh[v]
                else:
                    for 0 <= v <= ind:
                        prob1 += rh[v]
                        prob2 += rh[v]
                    prob2 += rh[v+1]
                    probs[i,j] = linear_interp(thresh, vals[ind], vals[ind+1],
                        prob1, prob2)

            # Create Ensemble Probabilities Greater Than Envelope
            else:
                # This is the CDF of the GUMBEL DISTRIBUTION
                mean = 0
                diffs = 0
                # Get Mean
                for 0 <= v < kk:
                    mean += vals[v]
                mean = mean / kk
                # Get Standard Deviation
                for 0 <= v < kk:
                    diffs += (vals[v] - mean)**2
                sigma = (diffs/kk)**0.5
                beta = (sigma * SQRT6) / PI
                mu = mean + EULER*beta
                prob1 = cdf(thresh, mu, beta)
                prob2 = cdf(vals[-1], mu, beta)
                probs[i,j] = rh[-1] - ((prob1-prob2)/(1.0-prob2)*rh[-1])

            probs[i,j] *= 100.

    return probs



