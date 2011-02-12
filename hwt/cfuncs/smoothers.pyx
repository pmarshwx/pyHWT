import Cython
import numpy as np
cimport numpy as np
cimport cython


cdef extern from 'math.h':
    float exp(float x)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t




@cython.boundscheck(False)
@cython.cdivision(True)
def gaussian(np.ndarray[DTYPE_t, ndim=2] data, 
             float sig, 
             float dx,
             float factor):

    cdef unsigned int ulength = data.shape[0]
    cdef unsigned int vlength = data.shape[1]
    cdef unsigned int ng, nx, ny, nw
    cdef int jw, je, isouth, inorth, ngn
    cdef float sigma, sigmasq, distsq, sqng, amp
    cdef Py_ssize_t i, j, ii, jj, nxx, nyy
    cdef float PI=3.141592653589793

    cdef np.ndarray[DTYPE_t, ndim=2] frc_data = np.zeros([ulength, vlength], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] partweight = np.zeros([ulength*vlength], dtype=DTYPE)
    
    
    sigma = sig/dx
    sigmasq = sigma*sigma

    ng = int(factor * sigma)
    sqng = float(ng * ng)
    ngn = -1 * ng
    nx = 2*ng+1
    ny = 2*ng+1
    nw=0
    
    for nxx in range(ngn, ng+1):
        for nyy in range(ngn, ng+1):
            nw = nw+1
            distsq = float(nxx*nxx) + float(nyy*nyy)
            if distsq <= sqng:
                partweight[nw] = exp(-0.5*distsq/sigmasq) 
                
    for i in range(0,ulength):
        for j in range(0,vlength):
            #print i, j
            if data[i,j] > 0:
                amp = data[i,j] / (2*PI*sigmasq)
                jw=j-ng
                je=j+ng
                isouth=i-ng
                inorth=i+ng
                nw=0
                for ii in range(isouth,inorth):
                    for jj in range(jw, je):
                        nw += 1
                        if jj < 0 or jj >= vlength or ii < 0 or ii >= ulength:
                            continue
                        frc_data[ii,jj] = frc_data[ii,jj] + amp*partweight[nw]
                        #print amp, partweight[nw], frc_data[ii,jj]
                            
                            
    return frc_data
                    
