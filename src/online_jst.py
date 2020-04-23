import numpy as np
import numbers
import _jst
import logging

logger = logging.getLogger()

class OJST:
    def __init__(self, n_topics, senti_lex=None, n_senti=3, n_iter=1000, refresh=10, random_state=None, window_size=1, theta=0.5):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.n_senti = n_senti

        self.theta = theta
        self.refresh = refresh
        self.random_state = random_state
        self.senti_lex = senti_lex

        self.B = []
        # random numbers that are reused
        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024 ** 2 // 8)  # 1MiB of random variates

    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_senti = self.n_senti
        n_topics = self.n_topics
        n_iter = self.n_iter

        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_sentiments: {}".format(n_senti))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.nd = np.zeros(D, dtype=np.intc)
        self.ndl = np.zeros((D, n_senti), dtype=np.intc)
        self.ndlz = np.zeros((D, n_senti, n_topics), dtype=np.intc)
        self.nlzw = np.zeros((n_senti, n_topics, W), dtype=np.intc)
        self.nlz = np.zeros((n_senti, n_topics), dtype=np.intc)

        self.WS, self.DS = self.matrix_to_lists(X)
        self.ZS, self.LS = np.empty_like(self.WS, dtype=np.intc), np.empty_like(self.WS, dtype=np.intc)

        np.testing.assert_equal(N, len(self.WS))
        for i in range(N):
            w, d = self.WS[i], self.DS[i]
            z_new = i % n_topics
            l_new = i % n_senti
            self.ZS[i] = z_new
            self.LS[i] = l_new
            self.nd[d] += 1
            self.ndl[d, l_new] += 1
            self.ndlz[d, l_new, z_new] += 1
            self.nlzw[l_new, z_new, w] += 1
            self.nlz[l_new, z_new] += 1

        self.loglikelihoods_ = []
        print("Initialization done!")

    def fit(self, X, alpha=0, beta=0, gamma=0):
        """
        Fit model with X
        :return:
        """
        self.init_params(X, alpha, beta, gamma)

        for t, x in enumerate(X):
            D, W = x.shape
            if t == 0:  # initialize with prior
                pass
            else:   # set for aggragation
                pass
            self._fit(x)
            # self.loglikelihoods_train.append(self.ll)

            self.B.append(self.phi_lzw)

    def _fit(self, X):
        random_state = self.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X)
        for it in range(self.n_iter):
            random_state.shuffle(rands)

            if it % self.refresh == 0:
                logger.info("Train %d / %d epoch" % (it, self.n_iter))
            #     ll = self.loglikelihood()
            #     logger.info("<{}> log likelihood: {:.0f}".format(it, ll))

            self.sample_(rands)
        # self.ll = self.loglikelihood()
        # logger.info("<{}> log likelihood: {:.0f}".format(self.n_iter - 1, self.ll))
        # compute components
        self.phi_lzw = self.compute_phi_lzw()

        del self.WS
        del self.DS
        del self.LS
        del self.ZS
        return self

    def sample_(self, rands):
        """
        Call Cython
        :param rands:
        :return:
        """

        _jst._sample_topics(self.WS, self.DS, self.ZS, self.LS, self.nd, self.ndl, self.ndlz, self.nlzw, self.nlz,
                            self.alpha_lz, self.alphaSum_l, self.beta_lzw, self.betaSum_lz,
                            self.gamma, rands)

    def compute_phi_lzw(self):
        phi_lzw = (self.nlzw + self.beta_lzw).astype(float) / (self.nlz + self.betaSum_lz)[:, :, np.newaxis]
        return phi_lzw

    def compute_pi_dl(self):
        pi_dl = (self.ndl + self.gamma).astype(float) / (self.nd + self.gamma * self.n_senti)[:, np.newaxis]
        return pi_dl

    def compute_theta_dlz(self):
        theta_dlz = (self.ndlz + self.alpha_lz).astype(float) / (self.ndl + self.alphaSum_l)[:, :, np.newaxis]
        return theta_dlz

    # def print_words(self, prob, n_top_words=10):
    #     for k, beta_k in enumerate(prob):
    #         topic_words = [self.vocab_bow[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
    #         yield 'Topic {}: {}'.format(k, ' '.join(x.encode('utf-8') for x in topic_words))

    def loglikelihood(self):
        """
        Compute ppl
        :return:
        """
        pass

    def check_random_state(self, seed):
        if seed is None:
            # i.e., use existing RandomState
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError("{} cannot be used as a random seed.".format(seed))

    def matrix_to_lists(self, doc_word):
        """Convert a (sparse) matrix of counts into arrays of word and doc indices

        Parameters
        ----------
        doc_word : array or sparse matrix (D, V)
            document-term matrix of counts

        Returns
        -------
        (WS, DS) : tuple of two arrays
            WS[k] contains the kth word in the corpus
            DS[k] contains the document index for the kth word

        """
        if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
            logger.warning("all zero row in document-term matrix found")
        if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
            logger.warning("all zero column in document-term matrix found")
        sparse = True
        try:
            # if doc_word is a scipy sparse matrix
            doc_word = doc_word.copy().tolil()
        except AttributeError:
            sparse = False

        if sparse and not np.issubdtype(doc_word.dtype, int):
            raise ValueError("expected sparse matrix with integer values, found float values")

        ii, jj = np.nonzero(doc_word)
        if sparse:
            ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
        else:
            ss = doc_word[ii, jj]

        n_tokens = int(doc_word.sum())
        DS = np.repeat(ii, ss).astype(np.intc)
        WS = np.empty(n_tokens, dtype=np.intc)
        startidx = 0
        for i, cnt in enumerate(ss):
            cnt = int(cnt)
            WS[startidx:startidx + cnt] = jj[i]
            startidx += cnt
        return WS, DS

    def init_params(self, X, alpha, beta, gamma):
        avg_doc_len = np.sum(X[0]) / X[0].shape[0]
        if alpha == 0:
            alpha = avg_doc_len * 0.05 / (self.n_senti * self.n_topics)
        self.alpha_lz = np.full((self.n_senti, self.n_topics), alpha, dtype=np.float64)
        self.alphaSum_l = np.sum(self.alpha_lz, axis=1)
        if beta == 0:
            self.beta_lzw = np.full((self.n_senti, self.n_topics, X[0].shape[1]), 0.01, dtype=np.float64)
        else:
            self.beta_lzw = np.full((self.n_senti, self.n_topics, X[0].shape[1]), beta, dtype=np.float64)
        for wid, pl in self.senti_lex.items():
            if pl == 1: # pos
                self.beta_lzw[:,:, wid] *= np.array([0.05, 0.9, 0.05])[:, np.newaxis]
            elif pl == -1:  # neg
                self.beta_lzw[:,:, wid] *= np.array([0.05, 0.05, 0.9])[:, np.newaxis]
        self.betaSum_lz = np.sum(self.beta_lzw, axis=2)
        self.gamma = gamma
        if gamma == 0:
            self.gamma = avg_doc_len * 0.05 / self.n_senti
