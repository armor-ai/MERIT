cimport cython
from cython.operator cimport preincrement as inc, predecrement as dec, dereference as dref
from libc.stdlib cimport malloc, free
from libc.math cimport log, exp

cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin

cdef double exp_sum_up(double* arr, int length) nogil:
    cdef int i = 0
    cdef double r = 0
    while i < length:
        r += exp(arr[i])
        arr[i] = r
        inc(i)
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:] LS, int[:] nd, int[:,:] ndl, int[:,:,:] ndlz, int[:,:,:] nlzw, int[:,:] nlz,
    double[:,:] alpha_lz, double[:] alphaSum_l, double[:,:,:] beta_lzw, double[:,:] betaSum_lz, double gamma,
    double[:] rands):

    cdef int w, d, z, l, z_n, l_n
    cdef int i, ij, l_i, z_j
    cdef int N = WS.shape[0]
    cdef int L = ndl.shape[1]
    cdef int Z = nlz.shape[1]

    cdef int n_rand = rands.shape[0]
    cdef int sn, samp
    cdef double r, prob_cum
    cdef double* prob = <double*> malloc(L*Z * sizeof(double))

    with nogil:
        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]
            l = LS[i]

            dec(nd[d])
            dec(ndl[d,l])
            dec(ndlz[d,l,z])
            dec(nlzw[l,z,w])
            dec(nlz[l,z])

            prob_cum = 0
            for l_i in range(L):
                for z_j in range(Z):
                    ij = l_i * Z + z_j
                    prob_cum += (nlzw[l_i,z_j,w] + beta_lzw[l_i,z_j,w]) / (nlz[l_i,z_j] + betaSum_lz[l_i,z_j]) * \
                                (ndlz[d,l_i,z_j] + alpha_lz[l_i,z_j]) / (ndl[d,l_i] + alphaSum_l[l_i]) * \
                                (ndl[d,l_i] + gamma) / (nd[d] + gamma * L)
                    prob[ij] = prob_cum

            r = rands[i % n_rand] * prob_cum
            samp = searchsorted(prob, L*Z, r)
            l_n = samp / Z
            z_n = samp % Z

            LS[i] = l_n
            ZS[i] = z_n

            inc(nd[d])
            inc(ndl[d,l_n])
            inc(ndlz[d,l_n,z_n])
            inc(nlzw[l_n,z_n,w])
            inc(nlz[l_n,z_n])

        free(prob)
