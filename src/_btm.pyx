#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec, dereference as dref
from libc.stdlib cimport malloc, free

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

def _sample_topics(int[:] BS1, int[:] BS2, int[:] ZS, int[:,:] nzw, int[:] nz,
    double[:] alpha_z, double[:,:] beta_zw, double[:] betaSum_z, double[:] rands):

    cdef int w1, w2, z, l, z_n, l_n
    cdef int i, z_j
    cdef int B = BS1.shape[0]
    cdef int Z = nzw.shape[0]

    cdef int n_rand = rands.shape[0]
    cdef int sn, samp
    cdef double r, prob_cum
    cdef double* prob = <double*> malloc(Z * sizeof(double))

    with nogil:
        for i in range(B):
            w1 = BS1[i]
            w2 = BS2[i]
            z = ZS[i]

            dec(nzw[z, w1])
            dec(nzw[z, w2])
            dec(nz[z])

            prob_cum = 0
            for z_j in range(Z):
                prob_cum += (nzw[z_j, w1] + beta_zw[z_j, w1]) / (2 * nz[z_j] + betaSum_z[z_j]) * \
                            (nzw[z_j, w2] + beta_zw[z_j, w2]) / (2 * nz[z_j] + 1 + betaSum_z[z_j]) * \
                            (nz[z_j] + alpha_z[z_j])
                prob[z_j] = prob_cum

            r = rands[i % n_rand] * prob_cum
            z_n = searchsorted(prob, Z, r)

            ZS[i] = z_n

            inc(nzw[z_n][w1])
            inc(nzw[z_n][w2])
            inc(nz[z_n])

        free(prob)
