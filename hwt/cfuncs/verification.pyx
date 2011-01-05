import Cython
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

DTYPE4 = np.double
ctypedef np.double_t DTYPE4_t




@cython.boundscheck(False)
def reliability(np.ndarray[DTYPE2_t, ndim=2] fcst,
                np.ndarray[DTYPE2_t, ndim=2] obs,
                np.ndarray[DTYPE2_t, ndim=2] mask,
                unsigned int asize):

    cdef unsigned int ii = fcst.shape[0]
    cdef unsigned int jj = fcst.shape[1]
    cdef Py_ssize_t i, j
    
    cdef np.ndarray[DTYPE4_t, ndim=1] fhist = np.zeros(asize+1, dtype=DTYPE4)
    cdef np.ndarray[DTYPE4_t, ndim=1] ohist = np.zeros(asize+1, dtype=DTYPE4)

    cdef unsigned int iii, jjj
    
    for i in range(ii):
        for j in range(jj):
            if mask[i,j] == 0 or mask[i,j] == 9999:
                continue
            elif fcst[i,j] < 0:
                fhist[0] += 1
                ohist[0] += obs[i,j]
            else:
                fhist[fcst[i,j]+1] += 1
                ohist[fcst[i,j]+1] += obs[i,j]
            
            
    return (fhist, ohist)
    


@cython.boundscheck(False)
@cython.cdivision(True)
def getContingency(np.ndarray[DTYPE_t, ndim=2] fcst, 
                   np.ndarray[DTYPE_t, ndim=2] obs):

    cdef unsigned int ulength = fcst.shape[0]
    cdef unsigned int vlength = fcst.shape[1]
    cdef unsigned int a, b, c, d
    cdef Py_ssize_t i, j
    
    a = 0
    b = 0
    c = 0
    d = 0
    
    for i in range(0, ulength):
        for j in range(0,vlength):
            if fcst[i,j] == 1:
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