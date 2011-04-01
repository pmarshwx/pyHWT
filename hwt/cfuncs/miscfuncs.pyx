cimport cython
import numpy as np
cimport numpy as np


DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t



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