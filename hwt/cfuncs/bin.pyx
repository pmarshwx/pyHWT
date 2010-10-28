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