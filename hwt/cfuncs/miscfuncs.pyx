import Cython
import numpy as np
cimport cython
cimport numpy as np


DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t



@cython.boundscheck(False)
def ptype(np.ndarray[DTYPE32_t, ndim=2] cloud,
          np.ndarray[DTYPE32_t, ndim=2] rain,
          np.ndarray[DTYPE32_t, ndim=2] snow,
          np.ndarray[DTYPE32_t, ndim=2] cldice,
          np.ndarray[DTYPE32_t, ndim=2] graupel):

    cdef unsigned int ii = cloud.shape[0]
    cdef unsigned int jj = cloud.shape[1]
    cdef Py_ssize_t i, j
    
    cdef np.ndarray[DTYPE_t, ndim=2] ptype = np.zeros([ii, jj], dtype=DTYPE)
    
    for i in range(ii):
        for j in range(jj):
            # Is rain the largest
            if (rain[i,j] > cloud[i,j] and rain[i,j] > snow[i,j] and 
                rain[i,j] > cldice[i,j] and rain[i,j] > graupel[i,j]):
                    ptype[i,j] = 1
            
            # Is snow the largest
            elif (snow[i,j] > cloud[i,j] and snow[i,j] > rain[i,j] and 
                  snow[i,j] > cldice[i,j] and cloud[i,j] > graupel[i,j]):
                    ptype[i,j] = 2
            
            # Is graupel the largest
            elif (graupel[i,j] > cloud[i,j] and graupel[i,j] > rain[i,j] and 
                  graupel[i,j] > snow[i,j] and graupel[i,j] > cldice[i,j]):
                    ptype[i,j] = 3

            # Is cloud the largest
            elif (cloud[i,j] > rain[i,j] and cloud[i,j] > snow[i,j] and 
                  cloud[i,j] > cldice[i,j] and cloud[i,j] > graupel[i,j]):
                    ptype[i,j] = 5

            # Is cldice the largest
            elif (cldice[i,j] > cloud[i,j] and cldice[i,j] > rain[i,j] and 
                  cldice[i,j] > snow[i,j] and cldice[i,j] > graupel[i,j]):
                    ptype[i,j] = 6
                    
            # Is any of the rain, snow, grapuel equal
            elif (rain[i,j] == snow[i,j] or rain[i,j] == graupel[i,j] or 
                  snow[i,j] == graupel[i,j]):
                    ptype[i,j] = 4
                    
            # Is cloud equal to cldice
            elif (cloud[i,j] == cldice[i,j]):
                    ptype[i,j] = 7
            
            # If nothing matches, set to 0
            else:
                    ptype[i,j] = 0
    
    return ptype