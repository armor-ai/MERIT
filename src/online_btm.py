import numpy as np
import numbers
import _btm
import logging

logger = logging.getLogger()

class OBTM:
    def __init__(self, n_topics, vocab_size, n_iter=1000, refresh=10, random_state=None, window_size=1, theta=0.5):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.theta = theta
        self.refresh = refresh
        self.random_state = random_state

        self.B = []
        # random numbers that are reused
        rng = self.check_random_state(random_state)
        self._rands = rng.rand(1024 ** 2 // 8)  # 1MiB of random variates

    def _initialize(self, X):
        B = len(X)
        W = self.vocab_size
        n_topics = self.n_topics
        n_iter = self.n_iter

        logger.info("n_biterms: {}".format(B))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        self.nzw = np.zeros((n_topics, W), dtype=np.intc)
        self.nz = np.zeros(n_topics, dtype=np.intc)

        self.BS1, self.BS2 = self.matrix_to_lists(X)
        self.ZS = np.empty_like(self.BS1, dtype=np.intc)

        for i in range(B):
            b1, b2 = self.BS1[i], self.BS2[i]
            z_new = i % n_topics
            self.ZS[i] = z_new
            self.nzw[z_new, b1] += 1
            self.nzw[z_new, b2] += 1
            self.nz[z_new] += 1

        self.loglikelihoods_ = []
        print("Initialization done!")

    def fit(self, X, alpha=0.1, eta=0.01):
        """
        Fit model with X
        :return:
        """
        # self.init_params(X, alpha, beta)

        for t, x in enumerate(X):
            if t == 0:  # initialize with prior
                self.init_params(X, alpha, eta)
                eta_m = np.full((self.n_topics, self.vocab_size), eta).astype(np.float64)
            else:   # set for aggragation
                eta_m = self.soft_align(self.B, self.window_size, self.theta).astype(np.float64)
                self.beta_zw = eta_m
                self.betaSum_z = np.sum(eta_m, 1)
            self.eta_l = eta_m
            self._fit(x)
            # self.loglikelihoods_train.append(self.ll)

            self.B.append(self.phi_zw)

    def soft_align(self, B, window_size, theta):
        """
        Soft alignment to produce a soft weight sum of B according to window size
        """
        eta = B[-1]
        eta_new = np.zeros(eta.shape)
        weights = self.softmax(eta, B, window_size)
        for i in range(window_size):
            if i > len(B)-1:
                break
            B_i = B[-i-1] * weights[i][:, np.newaxis]
            eta_new += B_i
        eta_new = theta * self.eta_l + (1 - theta) * eta_new
        return eta_new

    def softmax(self, eta, B, window_size):
        prods = []
        for i in range(window_size):
            if i > len(B)-1:
                break
            prods.append(np.einsum('ij,ij->i', eta, B[-i-1]))
        weights = np.exp(np.array(prods))
        # weights = np.ones(weights.shape)            # compare to uniform
        n_weights = weights / np.sum(weights, 0)  # column normalize
        return n_weights

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
        self.phi_zw = self.compute_phi_zw()

        del self.BS1
        del self.BS2
        del self.ZS
        return self

    def sample_(self, rands):
        """
        Call Cython
        :param rands:
        :return:
        """

        _btm._sample_topics(self.BS1, self.BS2, self.ZS, self.nzw, self.nz,
                            self.alpha_z, self.beta_zw, self.betaSum_z, rands)

    def compute_phi_zw(self):
        phi_zw = (self.nzw + self.beta_zw).astype(float) / (self.nz + self.betaSum_z)[:, np.newaxis]
        return phi_zw


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

    def matrix_to_lists(self, X):
        """Convert a (sparse) matrix of counts into arrays of biterms

        Parameters
        ----------
        X : array or biterms

        Returns
        -------
        BS1 : contains the first word in kth biterm in the corpus
        BS2 : contains the second word in kth biterm in the corpus

        """
        BS1 = []
        BS2 = []
        for bs in X:
            BS1.append(bs[0])
            BS2.append(bs[1])
        return np.array(BS1, dtype=np.intc), np.array(BS2, dtype=np.intc)


    def init_params(self, X, alpha, beta):
        avg_doc_len = 2
        if alpha == 0:
            alpha = 50.0 / self.n_topics
        self.alpha_z = np.full(self.n_topics, alpha, dtype=np.float64)
        self.alphaSum = np.sum(self.alpha_z)
        if beta == 0:
            self.beta_zw = np.full((self.n_topics, self.vocab_size), 0.01, dtype=np.float64)
        else:
            self.beta_zw = np.full((self.n_topics, self.vocab_size), beta, dtype=np.float64)
        self.betaSum_z = np.sum(self.beta_zw, axis=1)
