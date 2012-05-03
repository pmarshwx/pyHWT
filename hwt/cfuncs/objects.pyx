cimport cython
import numpy as np
cimport numpy as np

# cython: profile=True


DTYPE = np.int
DTYPE2 = np.double
DTYPE32 = np.float32
DTYPE64 = np.float64
ctypedef np.int_t DTYPE_t
ctypedef np.double_t DTYPE2_t
ctypedef np.float32_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t


@cython.boundscheck(False)
@cython.cdivision(True)
def create_object_3d(np.ndarray[DTYPE_t, ndim=3] mask,
    unsigned int t, unsigned int i, unsigned int j,
    unsigned int dt, unsigned int dx, unsigned int dy, int tol):

    cdef int max_t = 0
    cdef unsigned int tpt, ipt, jpt,
    cdef int tt, ii, jj
    cdef int ttt, iii, jjj
    cdef set inds = set()
    cdef list stack = []
    stack.append((t,i,j))
    while stack:
        tpt, ipt, jpt = stack.pop()
        if (tpt, ipt, jpt) in inds: continue
        inds.add((tpt, ipt, jpt))
        max_t = max(max_t, tpt)
        for tt in range(-tol, tol+1, 1):
            for ii in range(-tol, tol+1, 1):
                for jj in range(-tol, tol+1, 1):
                    ttt = tpt + tt
                    iii = ipt + ii
                    jjj = jpt + jj
                    if ttt < 0 or ttt >= dt: continue
                    if iii < 0 or iii >= dx: continue
                    if jjj < 0 or jjj >= dy: continue
                    if mask[ttt, iii, jjj] == 1: continue
                    mask[ttt, iii, jjj] = 1
                    stack.append((ttt, iii, jjj))
    return inds, max_t - t + 1


@cython.boundscheck(False)
@cython.cdivision(True)
def clark_3d_object(np.ndarray[DTYPE64_t, ndim=3] field,
    float min_thresh, float max_thresh, unsigned int min_t, unsigned int tol):
    cdef unsigned int dt, dx, dy, obj_counter
    cdef int t, i, j
    cdef list obj_start_time, all_inds

    dt = field.shape[0]
    dx = field.shape[1]
    dy = field.shape[2]
    field[field < min_thresh] = 0
    field[field > max_thresh] = 0
    field[field != 0] = 1

    cdef np.ndarray[DTYPE_t, ndim=3] field_obj = np.zeros([dt, dx, dy],
        dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] mask = np.zeros([dt, dx, dy],
        dtype=DTYPE)
    mask[field == 0] = 1
    obj_counter = 0
    obj_start_time = []
    all_inds = []
    for t in range(dt):
        for j in range(dy):
            for i in range(dx):
                if mask[t, i, j] == 1: continue
                inds, max_t = create_object_3d(mask, t, i, j, dt, dx, dy, tol)
                if max_t >= min_t:
                    all_inds.append(inds)
                    obj_counter += 1
                    obj_start_time.append(t)
                    for ind in inds:
                        field_obj[ind] = obj_counter
    return field_obj, obj_start_time, all_inds


@cython.boundscheck(False)
@cython.cdivision(True)
def define_ci(np.ndarray[DTYPE64_t, ndim=3] field,
    np.ndarray[DTYPE_t, ndim=3] field_obj, list pts,
    int radius):
    cdef unsigned int dt, dx, dy, dp, test
    cdef int ii, jj, tt, tpt, ipt, jpt
    cdef set loc
    dt = field.shape[0]
    dx = field.shape[1]
    dy = field.shape[2]
    dp = len(pts)
    cdef np.ndarray[DTYPE_t, ndim=3] field_init = np.zeros([dt, dx, dy],
        dtype=DTYPE)
    for i in range(dp):
        loc = set()
        dpp = len(pts[i])
        for j in range(dpp):
            test = 0
            tt = -1
            for jj in range(-radius, radius+1):
                for ii in range(-radius, radius+1):
                    tpt = pts[i][j][0] + tt
                    ipt = pts[i][j][1] + ii
                    jpt = pts[i][j][2] + jj
                    if tpt < 0 or tpt >= dt: continue
                    if ipt < 0 or ipt >= dx: continue
                    if jpt < 0 or jpt >= dy: continue
                    if field_obj[tpt, ipt, jpt] == i+1:
                        if field[tpt, ipt, jpt] <= field[pts[i][j]]:
                            test = 1
            if test == 0:
                # if pts[i][j][0] != 0:
                    loc.add(pts[i][j])
        for l in loc:
            field_init[l] = i+1
    return field_init


@cython.boundscheck(False)
@cython.cdivision(True)
def create_object_2d(np.ndarray[DTYPE_t, ndim=2] mask, unsigned int i,
    unsigned int j, unsigned int dx, unsigned int dy, unsigned int tol):
    cdef unsigned int ipt, jpt,
    cdef int ii, jj, iii, jjj
    cdef set inds = set()
    cdef list stack = []
    cdef list min_pts = []
    inds = set()
    stack.append((i,j))
    while stack:
        ipt, jpt = stack.pop()
        if (ipt, jpt) in inds: continue
        inds.add((ipt, jpt))
        for ii in range(-tol, tol+1, 1):
            for jj in range(-tol, tol+1, 1):
                iii = ipt + ii
                jjj = jpt + jj
                if iii < 0 or iii >= dx: continue
                if jjj < 0 or jjj >= dy: continue
                if mask[iii, jjj] == 1: continue
                mask[iii, jjj] = 1
                stack.append((iii, jjj))
    return inds


@cython.boundscheck(False)
@cython.cdivision(True)
def clark_2d_object(np.ndarray[DTYPE_t, ndim=2] field, float min_thresh,
    float max_thresh, unsigned int counter, unsigned int tol):
    cdef unsigned int dx, dy, tmp_count
    cdef long ipt, jpt
    cdef int i, j, k
    cdef float ii, jj
    cdef list obj_start_time, all_inds
    cdef list ipts, jpts, imeans, jmeans
    dx = field.shape[0]
    dy = field.shape[1]
    field[field < min_thresh] = 0
    if max_thresh != -1:
        field[field > max_thresh] = 0
    field[field != 0] = 1
    cdef np.ndarray[dtype=DTYPE_t, ndim=2] field_obj = np.zeros([dx, dy],
        dtype=DTYPE)
    cdef np.ndarray[dtype=DTYPE_t, ndim=2] mask = np.zeros([dx, dy],
        dtype=DTYPE)
    mask[field == 0] = 1
    imeans = []
    jmeans = []
    for j in range(dy):
        for i in range(dx):
            if mask[i, j] == 1: continue
            inds = create_object_2d(mask, i, j, dx, dy, tol)
            counter += 1
            ipts = []
            jpts = []
            temp_count = 0
            for ind in inds:
                ipt, jpt = ind
                ipts.append(ipt)
                jpts.append(jpt)
                temp_count += 1
                field_obj[ind] = counter
            imeans.append(sum(ipts) / temp_count)
            jmeans.append(sum(jpts) / temp_count)
    return field_obj, imeans, jmeans, counter











