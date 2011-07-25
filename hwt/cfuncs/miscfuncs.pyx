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
def geo_grid_data(np.ndarray[DTYPE64_t, ndim=1] ilon,
                  np.ndarray[DTYPE64_t, ndim=1] ilat,
                  np.ndarray[DTYPE64_t, ndim=2] mlons,
                  np.ndarray[DTYPE64_t, ndim=2] mlats,
                  float dx):

    cdef float PI = 3.14159265
    cdef float RADIUS = 3956.0
    cdef float PI_4_DEG2RAD = PI/180.0
    cdef float PI_4_RAD2DEG = 180.0/PI
    cdef float NM2KM = 69.0467669*1.609344

    cdef unsigned int kk = ilon.shape[0]
    cdef unsigned int jj = mlons.shape[1]
    cdef unsigned int ii = mlons.shape[0]
    cdef float min_dist, hdx
    cdef float c, x
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE64_t, ndim=2] rlat = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] dist = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] grid = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=1] rilat = np.zeros([kk], dtype=DTYPE64)
    cdef np.ndarray[DTYPE_t, ndim=1] xinds = np.zeros([kk], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yinds = np.zeros([kk], dtype=DTYPE)

    hdx = dx / 2.0
    for k in range(kk):
        rilat[k] = ilat[k] * PI_4_DEG2RAD

    for j in range(jj):
        for i in range(ii):
            rlat[i,j] = mlats[i,j] * PI_4_DEG2RAD

    for k in range(kk):
        if k % 50 == 0: print k
        min_dist = 99999.0
        for i in range(ii):
            for j in range(jj):
                c = (ilon[k]-mlons[i,j]) * PI_4_DEG2RAD
                x = (sinf(rlat[i,j]) * sinf(rilat[k]) + cosf(rlat[i,j]) *
                        cosf(rilat[k]) * cosf(c))
                dist[i,j] = acosf(x) * PI_4_RAD2DEG * NM2KM
                if dist[i,j] < min_dist:
                    min_dist = dist[i,j]
                    xinds[k] = i
                    yinds[k] = j
                    if min_dist <= hdx:
                        break
            if min_dist <= hdx:
                break
        grid[xinds[k], yinds[k]] += 1

    return (grid, xinds, yinds)


@cython.boundscheck(False)
def grid_data(np.ndarray[DTYPE64_t, ndim=1] xvals,
              np.ndarray[DTYPE64_t, ndim=1] yvals,
              np.ndarray[DTYPE64_t, ndim=2] xpts,
              np.ndarray[DTYPE64_t, ndim=2] ypts,
              float dx=1.):

    cdef unsigned int kk = xvals.shape[0]
    cdef unsigned int jj = xpts.shape[1]
    cdef unsigned int ii = ypts.shape[0]
    cdef float min_dist, hdx
    cdef Py_ssize_t i, j, k

    cdef np.ndarray[DTYPE64_t, ndim=2] dist = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] grid = np.zeros([ii,jj], dtype=DTYPE64)
    cdef np.ndarray[DTYPE_t, ndim=1] xinds = np.zeros([kk], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] yinds = np.zeros([kk], dtype=DTYPE)

    hdx = dx / 2.0
    for k in range(kk):
        if k % 1000 == 0: print k
        min_dist = 99999.0
        for i in range(ii):
            for j in range(jj):
                dist[i,j] = (xvals[k]-xpts[i,j])**2 + (yvals[k]-ypts[i,j])**2
                dist[i,j] = dist[i,j]**0.5
                if dist[i,j] < min_dist:
                    min_dist = dist[i,j]
                    xinds[k] = i
                    yinds[k] = j
                    if min_dist <= hdx:
                        break
            if min_dist <= hdx:
                break
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


@cython.boundscheck(False)
def layer_sum(np.ndarray[DTYPE32_t, ndim=4] uhfull,
              np.ndarray[DTYPE32_t, ndim=4] z,
              float zbot = 2000., float ztop = 5000.):

    cdef unsigned int kk = uhfull.shape[0]
    cdef unsigned int levs = uhfull.shape[1]
    cdef unsigned int jj = uhfull.shape[2]
    cdef unsigned int ii = uhfull.shape[3]

    cdef float btop, bbot, btmp
    cdef float ttop, tbot, ttmp
    cdef float tnm1, tnm2, tnm3
    cdef float bnm1, bnm2, bnm3
    cdef float bval, tval

    cdef np.ndarray[DTYPE32_t, ndim=3] uh = np.zeros([kk,jj,ii], dtype=DTYPE32)
    cdef Py_ssize_t bbptr, btptr
    cdef Py_ssize_t tbptr, ttptr
    cdef Py_ssize_t k, j, i, lev

    for k in range(kk):
        if k % 10 == 0: print k
        for j in range(jj):
            for i in range(ii):
                # Find nearest indices
                btop = 9999; ttop = 9999
                bbot = -9999; tbot = -9999
                btmp = -9999; ttmp = -9999
                for lev in range(levs):
                    btmp = z[k, lev, j, i] - zbot
                    ttmp = z[k, lev, j, i] - ztop
                    # Find pointers for bottom level
                    if btmp < 0:
                        if btmp > bbot:
                            bbot = btmp
                            bbptr = lev
                    elif btmp > 0:
                        if btmp < btop:
                            btop = btmp
                            btptr = lev
                    else:
                        bbot = btmp
                        btop = btmp
                        bbptr = lev
                        btptr = lev

                    # Find pointers for top level
                    if ttmp < 0:
                        if ttmp > tbot:
                            tbot = ttmp
                            tbptr = lev
                    elif ttmp > 0:
                        if ttmp < ttop:
                            ttop = ttmp
                            ttptr = lev
                    else:
                        tbot = ttmp
                        ttop = ttmp
                        tbptr = lev
                        ttptr = lev

                # Do bottom Interpolations
                if bbptr == btptr:
                    bval = uhfull[k,bbptr,j,i]
                else:
                    bbot = uhfull[k,bbptr,j,i]
                    btop = uhfull[k,btptr,j,i]
                    bnm1 = zbot - z[k,bbptr,j,i]
                    bnm2 = z[k,btptr,j,i] - z[k,bbptr,j,i]
                    bnm3 = bnm1 / bnm2
                    bval = bbot + bnm3 * (btop - bbot)

                # Do top Interpolations
                if tbptr == ttptr:
                    tval = uhfull[k,tbptr,j,i]
                else:
                    tbot = uhfull[k,tbptr,j,i]
                    ttop = uhfull[k,ttptr,j,i]
                    tnm1 = ztop - z[k,tbptr,j,i]
                    tnm2 = z[k,ttptr,j,i] - z[k,tbptr,j,i]
                    tnm3 = tnm1 / tnm2
                    tval = tbot + tnm3 * (ttop - tbot)


                for lev in range(btptr, ttptr+1):
                    if lev == btptr:
                        uh[k,j,i] += 0.5 * (bval + uhfull[k,lev,j,i]) * (z[k,lev,j,i] - zbot)
                    elif lev == ttptr:
                        uh[k,j,i] += 0.5 * (tval + uhfull[k,lev,j,i]) * (ztop - z[k,lev,j,i])
                    else:
                        uh[k,j,i] += (0.5 * (uhfull[k,lev,j,i]+uhfull[k,lev+1,j,i]) *
                                     (z[k,lev+1,j,i]-z[k,lev,j,i]))
                uh[k,j,i] += tval
    return uh




