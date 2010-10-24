import Cython
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

DTYPE3 = np.float64
ctypedef np.float64_t DTYPE3_t



@cython.boundscheck(False)
@cython.cdivision(True)
def circle(np.ndarray[DTYPE3_t, ndim=2] data, 
           float thresh, 
           float roi, 
           float dx):
    
    cdef unsigned int ulength = data.shape[0]
    cdef unsigned int vlength = data.shape[1]
    cdef unsigned int ng
    cdef int jw, je, isouth, inorth
    cdef float rng, distsq, dist
    cdef Py_ssize_t i, j, ii, jj

    cdef np.ndarray[DTYPE3_t, ndim=2] hitmiss = np.zeros([ulength, vlength], dtype=DTYPE3)
    cdef np.ndarray[DTYPE3_t, ndim=2] tmphit = np.zeros([ulength, vlength], dtype=DTYPE3)
    
    rng = roi/dx
    ng = int(rng)
    
    for j in range(vlength):
        for i in range(ulength):
            jw = j-ng
            je = j+ng + 1
            isouth = i-ng
            inorth = i+ng + 1
            if data[i,j] >= thresh:
                tmphit[i,j] = 1
                for jj in range(jw, je):
                    for ii in range(isouth, inorth):
                        distsq = float(j-jj)**2 + float(i-ii)**2
                        dist = distsq**0.5
                        if dist <= rng:
                            if jw < 0 or je >= vlength or isouth < 0 or inorth >= ulength:
                                continue
                            tmphit[ii,jj] = 1
                        
    return tmphit
    
   
@cython.boundscheck(False)
def findExceed(np.ndarray[DTYPE_t, ndim=2] var,
               np.ndarray[DTYPE2_t, ndim=2] mask,
               float thresh,
               int missing = 9999):

    cdef unsigned int ii = var.shape[0]
    cdef unsigned int jj = var.shape[1]
    cdef Py_ssize_t i, j
    
    for i in range(ii):
        for j in range(jj):
            if mask[i,j] == 0 or mask[i,j] == missing or var[i,j] == missing or var[i,j] < thresh:
                var[i,j] = 0
            else:
                var[i,j] = 1
                
    return(var)


@cython.boundscheck(False)
def findMaxGrid(np.ndarray[DTYPE_t, ndim=2] max,
                np.ndarray[DTYPE_t, ndim=2] var,
                np.ndarray[DTYPE2_t, ndim=2] mask,
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