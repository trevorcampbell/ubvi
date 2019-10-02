import autograd.numpy as np
from scipy.optimize import nnls 
from autograd.scipy.misc import logsumexp

from boostingvi import BoostingVI

class UBVI(BoostingVI):
    
    def __init__(self, logf, N, distribution, optimization, n_samples, n_init, n_logfg_samples, allow_negative=False):
        super().__init__(logf, N, distribution, optimization)
        self.n_samples = n_samples
        self.n_init = n_init
        self.init_inflation = 100
        self.n_logfg_samples = n_logfg_samples
        self.Z = np.zeros((self.N, self.N))
        self._logfg = -np.inf*np.ones(self.N)
        self._logfgsum = -np.inf
        self.allow_negative = allow_negative
        
        
    def _weights_update(self):
        i = self.params.shape[0]
        assert i > self.g_w.shape[0]
        Znew = np.exp(self.D.log_sqrt_pair_integral(self.params[i-1], self.params))
        self.Z[i-1, :i] = Znew
        self.Z[:i, i-1] = Znew
        self._logfg[i-1] = self._logfg_est(self.params[i-1])
        print('Updating weights...')
        if i == 1:
            return 1
        else:
            Linv = np.linalg.inv(np.linalg.cholesky(self.Z[:i, :i]))
            d = np.exp(self._logfg[:i]-self._logfg[:i].max()) #the opt is invariant to d scale, so normalize to have max 1
            b = nnls(Linv, -np.dot(Linv, d))[0]
            lbd = np.dot(Linv, b+d)
            return np.maximum(0., np.dot(Linv.T, lbd/np.sqrt(((lbd**2).sum()))))
        
        
    def _current_distance(self):
        i = self.params.shape[0]
        assert self.g_w.shape[0] == i and i > 0
        print('New weights: ' + str(self.g_w))
        self._logfgsum = logsumexp(np.hstack((-np.inf, self._logfg[:i] + np.log(np.maximum(self.g_w, 1e-64)))))
        hellsq = self._hellsq_update()
        print('New Hellinger-Squared estimate: ' + str(hellsq))
    
    
    def _objective(self, x, itr):
        '''
        Returns the negative of the objective function and should be minimized.
        '''
        assert self.g_w.shape[0] == self.params.shape[0]
        allow_negative = False if itr < 0 else True
        
        lgh = -np.inf if self.g_w.shape[0] == 0 else logsumexp(np.log(np.maximum(self.g_w, 1e-64)) + self.D.log_sqrt_pair_integral(x, self.params))
        h_samples = self.D.sample(x, self.n_samples)
        lf = self.target(h_samples)
        lh = 0.5 * self.D.logpdf(x, h_samples) 
        ln = np.log(self.n_samples)
        lf_num = logsumexp(lf - lh - ln)
        lg_num = self._logfgsum + lgh 
        log_denom = 0.5*np.log(1.-np.exp(2*lgh))
        if lf_num > lg_num:
            logobj = lf_num - log_denom + np.log(1.-np.exp(lg_num-lf_num))
            neglogobj = -logobj
            return neglogobj
        else:
            if not allow_negative:
                return np.inf
            lognegobj = lg_num - log_denom + np.log(1.-np.exp(lf_num-lg_num))
            return lognegobj


    def _hellsq_update(self):
        assert self.g_w.shape[0] == self.params.shape[0]
        samples = self._sample_g(self.n_samples)
        lf = self.target(samples)
        lg = self._logg(samples)
        ln = np.log(self.n_samples)
        return 1. - np.exp(logsumexp(lf-lg-ln) - 0.5*logsumexp(2*lf-2*lg-ln))


    def _logfg_est(self, param):
        samples = self.D.sample(param, self.n_samples)
        lf = self.target(samples)
        lg = 0.5 * self.D.logpdf(param, samples)
        ln = np.log(self.n_samples)
        return logsumexp(lf - lg - ln)
    
    
    def _logg(self, samples):
        i = self.params.shape[0]
        assert self.g_w.shape[0] == i and i > 0
        logg_x = 0.5 * self.D.logpdf(self.params, samples)
        if i==1:
            logg_x = logg_x[:,np.newaxis]
        return logsumexp(logg_x + np.log(np.maximum(self.g_w, 1e-64)), axis=1)
            
        
    def _sample_g(self, n):
        i = self.params.shape[0]
        assert self.g_w.shape[0] == i and i > 0
        #samples from g^2
        g_samples = np.zeros((n, self.D.d))
        #compute # samples in each mixture pair
        g_ps = (self.g_w[:, np.newaxis]*self.Z[:i, :i]*self.g_w).flatten()
        g_ps /= g_ps.sum()
        pair_samples = np.random.multinomial(n, g_ps).reshape((i, i))
        #symmetrize (will just use lower triangular below for efficiency)
        pair_samples = pair_samples + pair_samples.T
        for k in range(i):
            pair_samples[k,k] /= 2
        #invert sigs
        #fill in samples
        cur_idx = 0
        for j in range(i):
            for m in range(j+1):
                n_samps = pair_samples[j,m]
                g_samples[cur_idx:cur_idx+n_samps, :] = self.D.cross_sample(self.params[j], self.params[m], n_samps)
                cur_idx += n_samps
        return g_samples
       
       
        