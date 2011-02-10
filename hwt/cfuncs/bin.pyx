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
def precip(np.ndarray[DTYPE_t, ndim=2] var,
           np.ndarray[DTYPE_t, ndim=2] mask,
           unsigned int asize):

    cdef unsigned int ii = var.shape[0]
    cdef unsigned int jj = var.shape[1]
    cdef Py_ssize_t i, j
    
    cdef np.ndarray[DTYPE2_t, ndim=1] hist = np.zeros(asize, dtype=DTYPE2)

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
def joint_precip(np.ndarray[DTYPE_t, ndim=2] var1,
                 np.ndarray[DTYPE_t, ndim=2] var2,
                 np.ndarray[DTYPE_t, ndim=2] mask,
                 unsigned int asize):

    cdef unsigned int ii = var1.shape[0]
    cdef unsigned int jj = var1.shape[1]
    cdef Py_ssize_t i, j

    cdef np.ndarray[DTYPE2_t, ndim=1] hist1 = np.zeros(asize, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=1] hist2 = np.zeros(asize, dtype=DTYPE2)

    cdef unsigned int iii, jjj

    for i in range(ii):
        for j in range(jj):
            if (mask[i,j] == 0 or mask[i,j] == 9999 or 
                var1[i,j] == 9999 or var2[i,j]==9999):
                continue
            else:
                hist1[var1[i,j]] += 1
                hist2[var2[i,j]] += 1


    return (hist1, hist2)


@cython.boundscheck(False)
@cython.cdivision(True)
def regional_threshold(np.ndarray[DTYPE32_t, ndim=2] qpf, 
                       np.ndarray[DTYPE32_t, ndim=2] qpe,
                       np.ndarray[DTYPE_t, ndim=2] mask,
                       float roi, 
                       float dx,
                       float quantile,
                       int skip):
    
    cdef unsigned int ulength = qpf.shape[0]
    cdef unsigned int vlength = qpf.shape[1]
    cdef unsigned int ng, valid, ind
    cdef int missing = -9999
    cdef int jw, je, isouth, inorth
    cdef float rng, distsq, dist
    cdef Py_ssize_t i, j, ii, jj

    cdef np.ndarray[DTYPE32_t, ndim=2] qpfthreshes = np.zeros([ulength, vlength], dtype=DTYPE32)
    cdef np.ndarray[DTYPE32_t, ndim=2] qpethreshes = np.zeros([ulength, vlength], dtype=DTYPE32)
    cdef np.ndarray[DTYPE_t, ndim=2] validpts = np.zeros([ulength, vlength], dtype=DTYPE)
    
    rng = roi/dx
    ng = int(rng)
    
    if quantile > 1:
        quantile = quantile/100.
    
    for i from 0 <= i < ulength by skip:
        print i
        for j from 0 <= j < vlength by skip:
            valid = 0
            qpftmppts = []
            qpetmppts = []
            if mask[i,j] == 0 or mask[i,j] == 9999 or qpf[i,j] == 9999 or qpe[i,j] == 9999:
                if i+skip < ulength:
                    if j+skip < vlength:
                        #for ii from i <= ii < i+skip:
                        #    for jj from j <= jj < j+skip:
                        #        qpfthreshes[ii,jj] = missing
                        #        qpethreshes[ii,jj] = missing
                        qpfthreshes[i:i+skip,j:j+skip] = missing
                        qpethreshes[i:i+skip,j:j+skip] = missing
                    else:
                        #for ii from i <= ii < i+skip:
                        #    for jj from j <= jj < vlength:
                        #        qpfthreshes[ii,jj] = missing
                        #        qpethreshes[ii,jj] = missing
                        qpfthreshes[i:i+skip,j:] = missing
                        qpethreshes[i:i+skip,j:] = missing
                elif j+skip < vlength:
                    #for ii from i <= ii < ulength:
                    #    for jj from j <= jj < j+skip:
                    #        qpfthreshes[ii,jj] = missing
                    #        qpethreshes[ii,jj] = missing
                    qpfthreshes[i:,j:j+skip] = missing
                    qpethreshes[i:,j:j+skip] = missing
                else:
                    #for ii from i <= ii < ulength:
                    #    for jj from j <= jj < vlength:
                    #        qpfthreshes[ii,jj] = missing
                    #        qpethreshes[ii,jj] = missing
                    qpfthreshes[i:,j:] = missing
                    qpethreshes[i:,j:] = missing
                continue
            jw = j-ng
            je = j+ng + 1
            isouth = i-ng
            inorth = i+ng + 1
            for ii in range(isouth, inorth):
                for jj in range(jw, je):
                    if ii < 0 or ii >= ulength or jj < 0 or jj >= vlength:
                        continue
                    distsq = float(j-jj)**2 + float(i-ii)**2
                    dist = distsq**0.5
                    if dist <= rng:
                        if (mask[ii,jj] == 0 or mask[ii,jj] == 9999 or 
                           qpf[ii,jj] == 9999 or qpe[ii,jj] == 9999):
                            continue
                        else:
                            valid += 1
                            qpftmppts.append(qpf[ii,jj])
                            qpetmppts.append(qpe[ii,jj])

            ind = int(valid * quantile)
            qpftmp = np.array(qpftmppts)
            qpftmp = np.sort(qpftmp)
            qpetmp = np.array(qpetmppts)
            qpetmp = np.sort(qpetmp)
            if i+skip < ulength:
                if j+skip < vlength:
                    #for ii from i <= ii < i+skip:
                    #    for jj from j <= jj < j+skip:
                    #        qpfthreshes[ii,jj] = qpftmp[ind]
                    #        qpethreshes[ii,jj] = qpetmp[ind]
                    #        validpts[ii,jj] = valid
                    qpfthreshes[i:i+skip,j:j+skip] = qpftmp[ind]
                    qpethreshes[i:i+skip,j:j+skip] = qpetmp[ind]
                    validpts[i:i+skip,j:j+skip] = valid
                else:
                    #for ii from i <= ii < i+skip:
                    #    for jj from j <= jj < vlength:
                    #        qpfthreshes[ii,jj] = qpftmp[ind]
                    #        qpethreshes[ii,jj] = qpetmp[ind]
                    #        validpts[ii,jj] = valid
                    qpfthreshes[i:i+skip,j:] = qpftmp[ind]
                    qpethreshes[i:i+skip,j:] = qpetmp[ind]
                    validpts[i:i+skip,j:] = valid
            elif j+skip < vlength:
                #for ii from i <= ii < ulength:
                #    for jj from j <= jj < j+skip:
                #        qpfthreshes[ii,jj] = qpftmp[ind]
                #        qpethreshes[ii,jj] = qpetmp[ind]
                #        validpts[ii,jj] = valid
                qpfthreshes[i:,j:j+skip] = qpftmp[ind]
                qpethreshes[i:,j:j+skip] = qpetmp[ind]
                validpts[i:,j:j+skip] = valid
            else:
                #for ii from i <= ii < ulength:
                #    for jj from j <= jj < vlength:
                #        qpfthreshes[ii,jj] = qpftmp[ind]
                #        qpethreshes[ii,jj] = qpetmp[ind]
                #        validpts[ii,jj] = valid
                qpfthreshes[i:,j:] = qpftmp[ind]
                qpethreshes[i:,j:] = qpetmp[ind]
                validpts[i:,j:] = valid
                                
                        
    return (qpfthreshes, qpethreshes, validpts)





