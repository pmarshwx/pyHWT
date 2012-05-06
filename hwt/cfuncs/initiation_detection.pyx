cimport cython
import numpy as np
import sys
cimport numpy as np



cdef extern from 'math.h':
    float fabs(float x)


DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

DTYPE32 = np.float32
ctypedef np.float32_t DTYPE32_t

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t


cdef int MISSING = 0
cdef unsigned int NO_CONVECTION = 0
cdef unsigned int DECAYED_CONVECTION = 1
cdef unsigned int ONGOING_CONVECTION = 2
cdef unsigned int NEW_CONVECTION = 3


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef process(np.ndarray[DTYPE64_t, ndim=2] img0,
        np.ndarray[DTYPE64_t, ndim=2] img1, float object_threshold,
        int search_radius):
    cdef int dim0 = img0.shape[0]
    cdef int dim1 = img0.shape[1]
    aligned0 = _align_atob(img0, img1, dim0, dim1, object_threshold,
        search_radius)
    dilated0 = dilate(aligned0, dim0, dim1, 1.0, search_radius, 0,
        search_radius)
    ci = compute_ci(aligned0, dilated0, img1, dim0, dim1,
        object_threshold, search_radius)
    return ci


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef compute_warp_atob(np.ndarray[DTYPE64_t, ndim=2] a,
        np.ndarray[DTYPE64_t, ndim=2] b, int dim0, int dim1,
        float object_threshold, int search_radius):
    cdef int xblocksize = search_radius
    cdef int yblocksize = search_radius
    cdef int nxblocks = 1 + (dim0 / xblocksize)
    cdef int nyblocks = 1 + (dim1 / yblocksize)
    cdef int minx, maxx, nx, miny, maxy, ny
    cdef float uval, vval

    cdef np.ndarray[DTYPE64_t, ndim=2] u = np.zeros((nxblocks, nyblocks),
        dtype=DTYPE64)
    cdef np.ndarray[DTYPE64_t, ndim=2] v = np.zeros((nxblocks, nyblocks),
        dtype=DTYPE64)

    for 0 <= minx < dim0 by xblocksize:
        maxx = minx + xblocksize
        nx = minx / xblocksize
        for 0 <= miny < dim1 by yblocksize:
            maxy = miny + yblocksize
            ny = miny / yblocksize
            uval, vval = align_atob(a, b, dim0, dim1, object_threshold,
                search_radius, minx, maxx, miny, maxy)
            u[nx, ny] = uval
            v[nx, ny] = vval
    return u, v, xblocksize, yblocksize, nxblocks, nyblocks


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _align_atob(np.ndarray[DTYPE64_t, ndim=2] a,
        np.ndarray[DTYPE64_t, ndim=2] b, int dim0, int dim1,
        float object_threshold, int search_radius):
    warp = compute_warp_atob(a, b, dim0, dim1, object_threshold, search_radius)
    return align(a, dim0, dim1, warp)


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef align_atob(np.ndarray[DTYPE64_t, ndim=2] a,
        np.ndarray[DTYPE64_t, ndim=2] b, int dim0, int dim1,
        float object_threshold, int search_radius, int minx, int maxx,
        int miny, int maxy):
    # Maximizing overlap is the same as maximizing ongoing convection
    cdef float maxcsi = 0.1 # do not move echoes if resulting csi < this value
    cdef int maxoverlap = 3 # If less than 3 pixels of overlap, likely to be speckle
    cdef int setbyoverlap = 0
    cdef float u = 0
    cdef float v = 0
    cdef int m, n, i, j, nhits, nmiss, nfa, tot, calc_ok
    cdef float aval, bval, csi, min_err, err, n_err

    for m in range(-search_radius, search_radius+1):
        for n in range(-search_radius, search_radius+1):
            nhits = 0
            nmiss = 0
            nfa = 0
            for i in range(minx, maxx):
                for j in range(miny, maxy):
                    if i >= 0 and i < dim0 and i >= m and i < dim0+m and \
                       j >= 0 and j < dim1 and j >= n and j < dim1+n:
                        aval = a[i, j]
                        bval = b[i-m, j-n]
                        if aval > object_threshold:
                            if bval > object_threshold:
                                nhits += 1
                            else:
                                nmiss += 1
                        else:
                            nfa += 1
    # Overlap CSI
    tot = nhits + nmiss + nfa
    if tot > 0:
        csi = nhits / tot
        if nhits > maxoverlap:
            maxcsi = csi
            maxoverlap = nhits
            u = m
            v = n
            setbyoverlap = 1
    if setbyoverlap: return u, v
    # Overlap didn't work. Try minimizing mean absolute error.
    min_err = 3200
    for m in range(-search_radius, search_radius+1):
        for n in range(-search_radius, search_radius+1):
            err = 0
            n_err = 0
            calc_ok = 0
            for i in range(minx, maxx):
                for j in range(miny, maxy):
                    if i >= 0 and i < dim0 and i >= m and i < dim0+m and \
                       j >= 0 and j < dim1 and j >= n and j < dim1+n:
                        aval = a[i, j]
                        bval = b[i-m, j-n]
                        if aval == MISSING: aval = 0
                        else: calc_ok = 1
                        if bval == MISSING: bval = 0
                        else: calc_ok = 1
                        err += fabs(aval-bval)
                        n_err += 1
            if n_err > 1: err = err / n_err
            if calc_ok and err < min_err:
                min_err = err
                u = m
                v = n
    return u, v


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef align(np.ndarray[DTYPE64_t, ndim=2] a, int dim0, int dim1, tuple warp):
    cdef np.ndarray[DTYPE64_t, ndim=2] warp_u
    cdef np.ndarray[DTYPE64_t, ndim=2] warp_v
    cdef int warp_xblocksize, warp_yblocksize
    cdef int warp_nxblocks, warp_nyblocks
    cdef int maxdist, i, j, b1x, b1y, b2x, b2y, bx, byy, dx
    cdef int dy, distsq, sumu, sumv
    cdef int uinterp, vinterp
    cdef float sumwt, wt

    cdef np.ndarray[DTYPE64_t, ndim=2] aligned = np.zeros([dim0, dim1],
        dtype=DTYPE64)
    warp_u = warp[0]
    warp_v = warp[1]
    warp_xblocksize = warp[2]
    warp_yblocksize = warp[3]
    warp_nxblocks = warp[4]
    warp_nyblocks = warp[5]
    maxdist = dim0 * dim0 + dim1 * dim1
    cdef np.ndarray[DTYPE64_t, ndim=1] weights = np.zeros([maxdist-1],
        dtype=DTYPE64)
    weights = 1 / np.arange(1., maxdist)
    for i in range(0, dim0):
        for j in range(0, dim1):
            b1x = (i - warp_xblocksize / 2) / warp_xblocksize
            b1y = (j - warp_yblocksize / 2) / warp_yblocksize
            if b1x < 0: b1x = 0
            if b1y < 0: b1y = 0
            b2x = b1x + 1
            b2y = b1y + 1
            if b2x == warp_nxblocks: b2x = b1x
            if b2y == warp_nyblocks: b2y = b1y
            sumu = 0
            sumv = 0
            sumwt = 0.00001 # avoid divide-by-zero
            for bx in range(b1x, b2x+1):
                for byy in range(b1y, b2y):
                    dx = i - (bx * warp_xblocksize + warp_xblocksize / 2)
                    dy = j = (byy * warp_yblocksize + warp_yblocksize / 2)
                    distsq = dx*dx + dy*dy
                    wt = weights[distsq+1]
                    sumu += int(warp_u[bx, byy] * wt)
                    sumv += int(warp_v[bx, byy] * wt)
                    sumwt += wt
            uinterp = int(sumu / sumwt)
            vinterp = int(sumv / sumwt)
            if i >= -uinterp and i < dim0-uinterp and \
               j >= -vinterp and j < dim1-vinterp:
                aligned[i, j] = a[i+uinterp, j+vinterp]
            else:
                aligned[i, j] = MISSING
    return aligned


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef dilate(np.ndarray[DTYPE64_t, ndim=2] data, int dim0, int dim1,
        float percent, int halfx, int min_fill, int halfy):

    cdef np.ndarray[DTYPE64_t, ndim=2] temp = np.zeros((dim0, dim1),
        dtype=DTYPE64)
    cdef int i, j, m, n
    cdef int N
    cdef list values
    if percent > 1: percent = percent / 100.
    for i in range(halfx, dim0-halfx):
        for j in range(halfy, dim1-halfy):
            values = []
            for m in range(-halfx, halfx+1):
                for n in range(-halfy, halfy+1):
                    if data[i+m, j+n] != MISSING:
                        values.append(data[i+m, j+n])
            if len(values) > min_fill:
                N = int(0.5 + len(values) * percent)
                values.sort()
                if N >= len(values): N = len(values) - 1
                temp[i, j] = values[N]
    return temp


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef compute_ci(np.ndarray[DTYPE64_t, ndim=2] aligned0,
        np.ndarray[DTYPE64_t, ndim=2] dilated0,
        np.ndarray[DTYPE64_t, ndim=2] img1, int dim0, int dim1,
        float object_threshold, int search_radius):
    cdef int n_ongoing, n_decay, n_ci, i, j
    cdef list pixels
    n_ongoing = 0
    n_decay = 0
    n_ci = 0
    cdef np.ndarray[DTYPE64_t, ndim=2] ci = np.zeros((dim0, dim1),
        dtype=DTYPE64)
    for i in range(0, dim0):
        for j in range(0, dim1):
            is_convective = img1[i, j] > object_threshold
            was_near_convective = dilated0[i, j] > object_threshold
            was_convective = aligned0[i, j] > object_threshold
            if is_convective:
                if was_near_convective:
                    ci[i, j] = ONGOING_CONVECTION
                    n_ongoing += 1
                else:
                    ci[i, j] = NEW_CONVECTION
                    n_ci += 1
            else:
                if was_convective:
                    ci[i, j] = DECAYED_CONVECTION
                else:
                    ci[i, j] = NO_CONVECTION
    # Connected to a point of decayed convection, there cannot be
    # new convection. Change new to ongoing
    pixels = []
    for i in range(0, dim0):
        for j in range(0, dim1):
            if ci[i, j] == DECAYED_CONVECTION:
                pixels.append((i, j))
    ci = expand_decayed(ci, pixels, dim0, dim1, search_radius)
    # Connected to a point of ongoing convection, there cannot be
    # decayed or new convection. Change decayed to none;
    # new to ongoing.
    pixels = []
    for i in range(0, dim0):
        for j in range(0, dim1):
            if ci[i, j] == ONGOING_CONVECTION:
                pixels.append((i, j))
    ci = expand_ongoing(ci, pixels, dim0, dim1, search_radius)
    return ci


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef expand_decayed(np.ndarray[DTYPE64_t, ndim=2] ci, list pixels,
        int dim0, int dim1, int search_radius):
    cdef int x, y, m, n, i, j
    while pixels:
        x, y = pixels.pop()
        for m in range(-search_radius, search_radius+1):
            for n in range(-search_radius, search_radius+1):
                i = x + m
                j = y + n
                if i >= 0 and i < dim0 and j >= 0 and j < dim1:
                    if ci[i, j] == NEW_CONVECTION:
                        ci[i, j] = ONGOING_CONVECTION
                        pixels.append((i, j))
    return ci


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef expand_ongoing(np.ndarray[DTYPE64_t, ndim=2] ci, list pixels,
        int dim0, int dim1, int search_radius):
    cdef int x, y, m, n, i, j
    while pixels:
        x, y = pixels.pop()
        for m in range(-search_radius, search_radius+1):
            for n in range(-search_radius, search_radius+1):
                i = x + m
                j = y + n
                if i >= 0 and i < dim0 and j >= 0 and j < dim1:
                    if ci[i, j] == DECAYED_CONVECTION:
                        ci[i, j] = NO_CONVECTION
                        pixels.append((i, j))
                    elif ci[i, j] == NEW_CONVECTION:
                        ci[i, j] = ONGOING_CONVECTION
                        pixels.append((i, j))
    return ci


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef lak_ci(np.ndarray[DTYPE64_t, ndim=3] ca, float object_threshold,
        int search_radius):
    cdef unsigned int dt, dx, dy, t
    dt = ca.shape[0]
    dx = ca.shape[1]
    dy = ca.shape[2]
    cdef np.ndarray[DTYPE64_t, ndim=3] ci = np.zeros((dt, dx, dy),
        dtype=DTYPE64)
    for t in range(1, dt):
        ci[t, :, :] = process(ca[t-1, :, :], ca[t, :, :], object_threshold,
            search_radius)
    return ci
