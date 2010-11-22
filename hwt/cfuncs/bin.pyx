import Cython
import numpy as np
cimport numpy as np
cimport cython


DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

DTYPE4 = np.double
ctypedef np.double_t DTYPE4_t




@cython.boundscheck(False)
def precip(np.ndarray[DTYPE2_t, ndim=2] var,
           np.ndarray[DTYPE2_t, ndim=2] mask,
           unsigned int asize):

    cdef unsigned int ii = var.shape[0]
    cdef unsigned int jj = var.shape[1]
    cdef Py_ssize_t i, j
    
    cdef np.ndarray[DTYPE4_t, ndim=1] hist = np.zeros(asize, dtype=DTYPE4)

    cdef unsigned int max = 0
    cdef unsigned int iii, jjj
    
    for i in range(ii):
        for j in range(jj):
            if mask[i,j] == 0 or mask[i,j] == 9999 or var[i,j] == 9999:
                continue
            else:
                hist[var[i,j]] += 1
            
            
    return hist

    
@cython.boundscheck(False)
@cython.cdivision(True)
def regional_thresholds(np.ndarray[DTYPE_t, ndim=2] data, 
                        np.ndarray[DTYPE2_t, ndim=2] mask,
                        float roi, 
                        float dx):
    
    cdef unsigned int ulength = data.shape[0]
    cdef unsigned int vlength = data.shape[1]
    cdef unsigned int ng
    cdef int jw, je, isouth, inorth
    cdef float rng, distsq, dist
    cdef Py_ssize_t i, j, ii, jj

    cdef np.ndarray[DTYPE_t, ndim=2] hitmiss = np.zeros([ulength, vlength], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] tmphit = np.zeros([ulength, vlength], dtype=DTYPE)
    
    rng = roi/dx
    ng = int(rng)
    
    for j in range(vlength):
        for i in range(ulength):
            jw = j-ng
            je = j+ng + 1
            isouth = i-ng
            inorth = i+ng + 1
            for jj in range(jw, je):
                for ii in range(isouth, inorth):
                    distsq = float(j-jj)**2 + float(i-ii)**2
                    dist = distsq**0.5
                    if dist <= rng:
                        if jw < 0 or je >= vlength or isouth < 0 or inorth >= ulength:
                            continue
                        tmphit[ii,jj] = 1
                        
    return tmphit





