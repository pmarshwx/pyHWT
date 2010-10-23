import Cython
import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t




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



@cython.boundscheck(False)
@cython.cdivision(True)
def getRadiusContingency(np.ndarray[DTYPE_t, ndim=2] fcst, 
                         np.ndarray[DTYPE_t, ndim=2] obs,
                         float roi, 
                         float dx):

    cdef unsigned int ulength = fcst.shape[0]
    cdef unsigned int vlength = fcst.shape[1]
    cdef unsigned int a, b, c, d
    cdef unsigned int ng, skip
    cdef int jw, je, isouth, inorth
    cdef float rng, distsq, dist
    cdef Py_ssize_t i, j, ii, jj
    
    a = 0
    b = 0
    c = 0
    d = 0
    skip = 0

    rng = roi/dx
    ng = int(rng)
    
    for i in range(0, ulength):
        for j in range(0,vlength):
            if fcst[i,j] == 1:
                if obs[i,j] == 1:
                    a += 1
                else:
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
                                if obs[ii,jj] == 1:
                                    a+=1
                                    skip = 1
                                    break
                        if skip == 1:
                            break
                    if skip == 1:
                        skip = 0
                    else:
                        b += 1
            else:
                if obs[i,j] == 1:
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
                                if obs[ii,jj] == 1:
                                    skip = 1
                                    break
                        if skip == 1:
                            break
                    if skip == 1:
                        skip = 0
                    else:
                        c += 1
                else:
                    d += 1
                
    return (a, b, c, d)
