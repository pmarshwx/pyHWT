cimport cython
import numpy as np
cimport numpy as np


cdef extern from 'math.h':
    float exp(float x)
    float cos(float x)
    float sin(float x)

cdef float exp(float x):
    return exp(x)

cdef float cos(float x):
    return cos(x)
    
cdef float sin(float x):
    return sin(x)

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t




@cython.boundscheck(False)
@cython.cdivision(True)
def isotropic_gauss(np.ndarray[DTYPE64_t, ndim=2] data, 
                    float sig, 
                    float dx,
                    float factor):

    cdef unsigned int vlength = data.shape[0]
    cdef unsigned int ulength = data.shape[1]
    cdef unsigned int ng, nx, ny, nw
    cdef int iw, ie, js, jn, ngn
    cdef float sig_sq, dist_sq, ng_sq, amp
    cdef Py_ssize_t i, j, ii, jj, nxx, nyy
    cdef float PI=3.141592653589793

    cdef np.ndarray[DTYPE64_t, ndim=2] frc_data = np.zeros([vlength, ulength], dtype=DTYPE64)
        
    sig = sig/dx
    sig_sq = sig*sig

    ng = int(factor * sig)
    ng_sq = float(ng * ng)
    ngn = -1 * ng
    nx = 2*ng+1
    ny = 2*ng+1

    cdef np.ndarray[DTYPE64_t, ndim=1] partweight = np.zeros([nx*ny], dtype=DTYPE64)


    nw=-1
    for nyy in range(ngn, ng+1):
        for nxx in range(ngn, ng+1):
            nw = nw+1
            dist_sq = float(nxx*nxx) + float(nyy*nyy)
            if dist_sq <= ng_sq:
                partweight[nw] = exp(-0.5*dist_sq/sig_sq) 
                
    for j in range(0, vlength):
        for i in range(0, ulength):
            if data[j,i] > 0:
                amp = data[j,i] / (2*PI*sig_sq)
                iw=i-ng
                ie=i+ng
                js=j-ng
                jn=j+ng
                nw = -1
                for jj in range(js, jn+1):
                    for ii in range(iw, ie+1):
                        nw += 1
                        if jj < 0 or jj >= vlength or ii < 0 or ii >= ulength:
                            continue
                        frc_data[jj,ii] = frc_data[jj,ii] + amp*partweight[nw]

    return frc_data
                    

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef anisotropic_gauss(np.ndarray[DTYPE64_t, ndim=2] data, 
                        float sigx, float sigy, float rot, 
                        int h, int k, float dx, float factor, 
                        sig_as_grid_points=False):

    cdef unsigned int vlength = data.shape[0]
    cdef unsigned int ulength = data.shape[1]
    cdef unsigned int a, b, nxy, pwdimlength
    cdef int nxyn, iw, ie, js, jn, nw
    cdef float sigx_sq, sigy_sq, rad, amp
    cdef float sintheta, sintheta_sq, costheta, costheta_sq
    cdef float A, B, C, D, E, F, ellipse, X, Y
    cdef Py_ssize_t i, j, ii, jj, x, y
    
    cdef np.ndarray[DTYPE64_t, ndim=2] frc_data = np.zeros([vlength, ulength], dtype=DTYPE64)
    cdef float PI=3.141592653589793

    if not sig_as_grid_points:
        sigx = float(sigx / dx)
        sigy = float(sigy / dx)

    sigx_sq = sigx*sigx
    sigy_sq = sigy*sigy    

    # Set up X-axis
    a = int(factor * sigx)
    a_sq = float(a * a)

    # Set up Y-axis
    b = int(factor * sigy)
    b_sq = float(b * b)

    # Set weight-loop to size of maximum length
    if a >= b:
        nxy = a
    else: 
        nxy = b
    
    nxyn = -1 * nxy
    pwdimlength = 2*nxy+1

    # Rotation Matrix Values
    rad = rot * PI / 180. * -1.   # Not entirely sure why the -1, but it makes it work
    sintheta = sin(rad)
    sintheta_sq = sintheta**2
    costheta = cos(rad)
    costheta_sq = costheta**2

    # Constants used in transformation
    A = (costheta_sq / a_sq) + (sintheta_sq / b_sq)
    B = -2 * costheta * sintheta * ((1 / a_sq) - (1 / b_sq))
    C = (sintheta_sq / a_sq) + (costheta_sq / b_sq)
    # D = (2 * A * h ) + (B * k)
    # E = (2 * C * k) + (B * h)
    # F = (A * h * h) + (B * h * k) + (C * k *  k)

    cdef np.ndarray[DTYPE64_t, ndim=1] partweight = np.zeros([pwdimlength**2], dtype=DTYPE64)
    
    nw = -1
    for y in range(nxyn, nxy+1):
        for x in range(nxyn, nxy+1):
            nw += 1
            ellipse = (A*x*x) + (B*x*y) + (C*y*y) # - D*X) - (E*Y) + F
            if ellipse <= 1:
                X = (x)*costheta - (y)*sintheta
                Y = (x)*sintheta + (y)*costheta
                partweight[nw] = exp(-0.5*((X*X / sigx_sq) + (Y*Y / sigy_sq))) 

    for j in range(0, vlength):
        for i in range(0, ulength):
            if data[j,i] > 0:
                amp = data[j,i] / (2*PI*sigx*sigy)
                iw=i-nxy
                ie=i+nxy
                js=j-nxy
                jn=j+nxy
                nw = -1
                for jj in range(js+k, jn+k+1):
                    for ii in range(iw+h, ie+h+1):
                        nw += 1
                        if jj < 0 or jj >= vlength or ii < 0 or ii >= ulength:
                            continue
                        frc_data[jj,ii] = frc_data[jj,ii] + amp*partweight[nw]

    return frc_data
