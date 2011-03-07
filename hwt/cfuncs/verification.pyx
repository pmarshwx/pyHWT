import Cython
import numpy as np
cimport numpy as np
cimport cython


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

    cdef unsigned int iii, jjj
    
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

    cdef unsigned int ulength = fcst.shape[0]
    cdef unsigned int vlength = fcst.shape[1]
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