import Cython
import numpy as np
cimport numpy as np
cimport cython



DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t


@cython.boundscheck(False)
@cython.cdivision(True)
def circle(np.ndarray[DTYPE64_t, ndim=2] data, 
           float thresh, 
           float roi, 
           float dx):
    
    cdef unsigned int ulength = data.shape[0]
    cdef unsigned int vlength = data.shape[1]
    cdef unsigned int ng
    cdef int jw, je, isouth, inorth
    cdef float rng, distsq, dist
    cdef Py_ssize_t i, j, ii, jj

    cdef np.ndarray[DTYPE64_t, ndim=2] hitmiss = np.zeros([ulength, vlength], dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] tmphit = np.zeros([ulength, vlength], dtype=DTYPE64)
    
    rng = roi/dx
    ng = int(rng)
    
    for i in range(ulength):
        for j in range(vlength):
            jw = j-ng
            je = j+ng + 1
            isouth = i-ng
            inorth = i+ng + 1
            if data[i,j] >= thresh:
                tmphit[i,j] = 1
                for ii in range(isouth, inorth):
                    for jj in range(jw, je):
                        distsq = float(j-jj)**2 + float(i-ii)**2
                        dist = distsq**0.5
                        if dist <= rng:
                            if jw < 0 or je >= vlength or isouth < 0 or inorth >= ulength:
                                continue
                            tmphit[ii,jj] = 1
                        
    return tmphit
    
   
@cython.boundscheck(False)
@cython.cdivision(True)
def error_composite(np.ndarray[DTYPE_t, ndim=2] fcst,
                    np.ndarray[DTYPE_t, ndim=2] obs,
                    float radius, 
                    float dx):

    cdef unsigned int ulength = fcst.shape[0]
    cdef unsigned int vlength = fcst.shape[1]
    cdef unsigned int ng, nx, ny, nw
    cdef int jw, je, isouth, inorth, ngn
    cdef float distsq, sqng
    cdef Py_ssize_t i, j, ii, jj, nxx, nyy

    ng = int(radius / dx)
    sqng = float(ng * ng)
    nx = 2*ng+1
    ny = 2*ng+1

    cdef np.ndarray[DTYPE_t, ndim=2] errors = np.zeros([nx, ny], dtype=DTYPE)

    for i in range(0,ulength):
        for j in range(0,vlength):
            if fcst[i,j] > 0:
                for ii in range(-ng, ng+1):
                    for jj in range(-ng, ng+1):
                        iii = i + ii
                        jjj = j + jj
                        if jjj < 0 or jjj >= vlength or iii < 0 or iii >= ulength:
                            continue
                        elif obs[iii,jjj] > 0:
                            errors[ii,jj] += 1
                        else:
                            continue

    return errors


@cython.boundscheck(False)
def findExceed(np.ndarray[DTYPE64_t, ndim=2] var,
               np.ndarray[DTYPE_t, ndim=2] mask,
               float thresh,
               int missing = 9999):

    cdef unsigned int ii = var.shape[0]
    cdef unsigned int jj = var.shape[1]
    cdef Py_ssize_t i, j

    cdef np.ndarray[DTYPE64_t, ndim=2] newvar = np.zeros([ii, jj], dtype=DTYPE64)

    for i in range(ii):
        for j in range(jj):
            if mask[i,j] == 0 or mask[i,j] == missing or var[i,j] == missing or var[i,j] < thresh:
                newvar[i,j] = 0
            else:
                newvar[i,j] = 1

    return(newvar)


@cython.boundscheck(False)
def findRegionalExceed(np.ndarray[DTYPE64_t, ndim=2] var,
                       np.ndarray[DTYPE64_t, ndim=2] thresh,
                       np.ndarray[DTYPE_t, ndim=2] mask,
                       float minthresh = 25.4,
                       int missing = 9999):

    cdef unsigned int ii = var.shape[0]
    cdef unsigned int jj = var.shape[1]
    cdef Py_ssize_t i, j

    cdef np.ndarray[DTYPE64_t, ndim=2] newvar = np.zeros([ii, jj], dtype=DTYPE64)

    for i in range(ii):
        for j in range(jj):
            if (mask[i,j] == 0 or mask[i,j] == missing or 
                var[i,j] == missing or var[i,j] < thresh[i,j]):
                    newvar[i,j] = 0
            else:
                    if var[i,j] >= minthresh:
                        newvar[i,j] = 1
                    else:
                        newvar[i,j] = 0

    return(newvar)


@cython.boundscheck(False)
def findMaxGrid(np.ndarray[DTYPE64_t, ndim=2] max,
                np.ndarray[DTYPE64_t, ndim=2] var,
                np.ndarray[DTYPE_t, ndim=2] mask,
                int missing = 9999):

    cdef unsigned int ii = var.shape[0]
    cdef unsigned int jj = var.shape[1]
    cdef Py_ssize_t i, j
    
    for i in range(ii):
        for j in range(jj):
            if mask[i,j] == 0 or mask[i,j] == missing or var[i,j] == missing or var[i,j] < max[i,j]:
                var[i,j] = 0
            else:
                max[i,j] = var[i,j]
                
    return(max)