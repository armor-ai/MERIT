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

def _sample_topics(int[:] BS1, int[:] BS2, int[:] ZS, int[:] LS, int[:,:,:] nlzw, int[:,:] nlz, int[:] nl,
    double[:,:] alpha_lz, double[:] alphaSum_l, double[:,:,:] beta_lzw, double[:,:] betaSum_lz, double gamma,
    double[:] rands):

    cdef int w1, w2, z, l, z_n, l_n
    cdef int i, ij, l_i, z_j
    cdef int B = BS1.shape[0]
    cdef int L = nlz.shape[0]
    cdef int Z = nlz.shape[1]

    cdef int n_rand = rands.shape[0]
    cdef int sn, samp
    cdef double r, prob_cum
    cdef double* prob = <double*> malloc(L*Z * sizeof(double))

    with nogil:
        for i in range(B):
            w1 = BS1[i]
            w2 = BS2[i]
            z = ZS[i]
            l = LS[i]

            dec(nlzw[l,z,w1])
            dec(nlzw[l,z,w2])
            dec(nlz[l,z])
            dec(nl[l])

            prob_cum = 0
            for l_i in range(L):
                for z_j in range(Z):
                    ij = l_i * Z + z_j
                    prob_cum += (nlzw[l_i,z_j,w1] + beta_lzw[l_i,z_j,w1]) / (2 * nlz[l_i,z_j] + betaSum_lz[l_i,z_j]) * \
                                (nlzw[l_i,z_j,w2] + beta_lzw[l_i,z_j,w2]) / (2 * nlz[l_i,z_j] + 1 + betaSum_lz[l_i,z_j]) * \
                                (nlz[l_i,z_j] + alpha_lz[l_i,z_j]) / (nl[l_i] + alphaSum_l[l_i]) * \
                                (nl[l_i] + gamma) / (B + gamma * L)
                    prob[ij] = prob_cum

            r = rands[i % n_rand] * prob_cum
            samp = searchsorted(prob, L*Z, r)
            l_n = samp / Z
            z_n = samp % Z

            LS[i] = l_n
            ZS[i] = z_n

            inc(nlzw[l_n,z_n,w1])
            inc(nlzw[l_n,z_n,w2])
            inc(nlz[l_n,z_n])
            inc(nl[l_n])

        free(prob)
