cimport cython
import numpy as np
cimport numpy as np


DTYPE = np.int
DTYPE2 = np.double
DTYPE32 = np.float32
DTYPE64 = np.float64
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE2_t
ctypedef np.float32_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t

cdef extern from 'math.h':
    float sinf(float x)
    float cosf(float x)
    float acosf(float x)


@cython.boundscheck(False)
def grid_data(np.ndarray[DTYPE64_t, ndim=2] mlons,
              np.ndarray[DTYPE64_t, ndim=2] mlats,
              np.ndarray[DTYPE64_t, ndim=1] ilon,
              np.ndarray[DTYPE64_t, ndim=1] ilat):

    cdef float PI = 3.14159265
    cdef float RADIUS = 3956.0
    cdef float PI_4_DEG2RAD = PI/180.0
    cdef float PI_4_RAD2DEG = 180.0/PI
    cdef float NM2MI = 69.0467669

    cdef unsigned int kk = ilon.shape[0]
    cdef unsigned int jj = mlons.shape[1]
    cdef unsigned int ii = mlons.shape[0]
    cdef float min_dist
    cdef float c, x
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE64_t, ndim=2] rlat = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] dist = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] grid = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=1] rilat = np.zeros([kk], dtype=DTYPE64)
    cdef np.ndarray[DTYPE_t, ndim=1] xinds = np.zeros([kk], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yinds = np.zeros([kk], dtype=DTYPE)

    for k in range(kk):
        rilat[k] = ilat[k] * PI_4_DEG2RAD

    for j in range(jj):
        for i in range(ii):
            rlat[i,j] = mlats[i,j] * PI_4_DEG2RAD

    for k in range(kk):
        if k % 50 == 0: print k
        min_dist = 99999.0
        for j in range(jj):
            for i in range(ii):
                c = (ilon[k]-mlons[i,j]) * PI_4_DEG2RAD
                x = (sinf(rlat[i,j]) * sinf(rilat[k]) + cosf(rlat[i,j]) *
                        cosf(rilat[k]) * cosf(c))
                dist[i,j] = acosf(x) * PI_4_RAD2DEG * NM2MI
                if dist[i,j] < min_dist:
                    min_dist = dist[i,j]
                    xinds[k] = i
                    yinds[k] = j
        grid[xinds[k], yinds[k]] += 1

    return (grid, xinds, yinds)










@cython.boundscheck(False)
def ptype(np.ndarray[DTYPE2_t, ndim=2] rain,
          np.ndarray[DTYPE2_t, ndim=2] snow,
          np.ndarray[DTYPE2_t, ndim=2] graupel,
          np.ndarray[DTYPE2_t, ndim=2] cloud,
          np.ndarray[DTYPE2_t, ndim=2] ice,
          np.ndarray[DTYPE2_t, ndim=2] t2m,
          float minimum_threshold=0.01):

    cdef unsigned int ii = cloud.shape[0]
    cdef unsigned int jj = cloud.shape[1]
    cdef Py_ssize_t i, j

    cdef np.ndarray[DTYPE_t, ndim=2] ptype = np.zeros([ii, jj], dtype=DTYPE)

    for i in range(ii):
        for j in range(jj):
            # Is rain the largest
            if (rain[i,j] > cloud[i,j] and rain[i,j] > snow[i,j] and
                rain[i,j] > ice[i,j] and rain[i,j] > graupel[i,j]):
                    if t2m[i,j] > 273.15:
                        ptype[i,j] = 1
                    # Check for Freezing Rain
                    else:
                        ptype[i,j] = 5
            # Is snow the largest
            elif (snow[i,j] > cloud[i,j] and snow[i,j] > rain[i,j] and
                  snow[i,j] > ice[i,j] and snow[i,j] > graupel[i,j]):
                    ptype[i,j] = 2

            # Is graupel the largest
            elif (graupel[i,j] > cloud[i,j] and graupel[i,j] > rain[i,j] and
                  graupel[i,j] > snow[i,j] and graupel[i,j] > ice[i,j]):
                    ptype[i,j] = 3

            # Is cloud the largest
            elif (cloud[i,j] > rain[i,j] and cloud[i,j] > snow[i,j] and
                  cloud[i,j] > ice[i,j] and cloud[i,j] > graupel[i,j]):
                  if t2m[i,j] > 273.15:
                      ptype[i,j] = 6
                  # Check for Freezing Fog
                  else:
                      ptype[i,j] = 7

            # Is ice the largest
            elif (ice[i,j] > cloud[i,j] and ice[i,j] > rain[i,j] and
                  ice[i,j] > snow[i,j] and ice[i,j] > graupel[i,j]):
                    # If ice is the largest, make sure it's greater than
                    # minimum threshold
                    if ice[i,j] > minimum_threshold:
                        ptype[i,j] = 8
                    else:
                        continue

            # Is any of the rain, snow, grapuel equal
            elif (rain[i,j] == snow[i,j] or rain[i,j] == graupel[i,j] or
                  snow[i,j] == graupel[i,j]):
                    if (rain[i,j] == 0 and snow[i,j] == 0 and
                        graupel[i,j] == 0):
                            ptype[i,j] = 0
                    else:
                            ptype[i,j] = 4

            # Is cloud equal to ice
            elif (cloud[i,j] == ice[i,j]):
                    if (cloud[i,j] == 0 and ice[i,j] == 0):
                        ptype[i,j] = 0
                    else:
                        ptype[i,j] = 9

            # If nothing matches, skip
            else:
                    continue

    return ptype