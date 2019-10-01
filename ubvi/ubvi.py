import autograd.numpy as np
from scipy.optimize import nnls 
from autograd.scipy.misc import logsumexp

from boostingvi import BoostingVI

class UBVI(BoostingVI):
    
    def __init__(self, logf, N, distribution, optimization, n_samples, n_init, n_logfg_samples):
        super().__init__(logf, N, distribution, optimization)
        self.n_samples = n_samples
        self.n_init = n_init
        self.init_inflation = 100
        self.n_logfg_samples = n_logfg_samples
        self.Z = np.zeros((self.N, self.N))
        self._logfg = -np.inf*np.ones(self.N)
        self._logfgsum = -np.inf
        
    def _new_weights(self, i):
        Znew = np.exp(self.D.log_sqrt_pair_integral(self.params[i], self.params[:i+1]))
        self.Z[i, :i+1] = Znew
        self.Z[:i+1, i] = Znew
        self._logfg[i] = self._logfg_est(i)
        print('Updating weights...')
        if i == 0:
            return 1
        else:
            Linv = np.linalg.inv(np.linalg.cholesky(self.Z[:i+1, :i+1]))
            d = np.exp(self._logfg[:i+1]-self._logfg[:i+1].max()) #the opt is invariant to d scale, so normalize to have max 1
            b = nnls(Linv, -np.dot(Linv, d))[0]
            lbd = np.dot(Linv, b+d)
            return np.maximum(0., np.dot(Linv.T, lbd/np.sqrt(((lbd**2).sum()))))
        
    def _current_distance(self, i):
        print('New weights: ' + str(self.g_w[:i+1]))
        #compute current log<f, g> estimate
        self._logfgsum = logsumexp(np.hstack((-np.inf, self._logfg[:i+1] + np.log(np.maximum(self.g_w[:i+1], 1e-64)))))
        hellsq = self._hellsq_est(i)
        print('New Hellinger-Squared estimate: ' + str(hellsq))
    
    def _objective(self, x, Params, W):
        '''
        Returns the negative of the objective function and should be minimized.
        '''
        lgh = -np.inf if W.shape[0] == 0 else logsumexp(np.log(np.maximum(W, 1e-64)) + self.D.log_sqrt_pair_integral(x, Params))
        #get samples from h
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
            lognegobj = lg_num - log_denom + np.log(1.-np.exp(lf_num-lg_num))
            return lognegobj
    
    def _objective_old(self, x, Params, W):
        if W.shape[0] > 0:
            log_denom = 0.5*logsumexp( np.array( [0., 2.*logsumexp(np.log(np.maximum(W, 1e-64)) + self.D.log_sqrt_pair_integral(x, Params))] ), b=np.array([1., -1.]))
        else:
            log_denom = 0.
        h_samples = self.D.sample(x, self.n_samples)
        lh = 0.5 * self.D.logpdf(x, h_samples)
        lf = self.target(h_samples)
        lg = self._logg(h_samples, W.shape[0]) + self._logfgsum
        ln = np.log(self.n_samples)
        #return objective
        sgns = np.hstack((np.ones(self.n_samples), -1.*np.ones(self.n_samples)))
        if logsumexp(lf - lh - ln) > logsumexp(lg - lh - ln):
            logobj = logsumexp( np.hstack(( lf - lh - ln, lg - lh - ln )), b=sgns ) - log_denom
            neglogobj = -logobj
            return neglogobj
        else:
            lognegobj = logsumexp( np.hstack(( lf - lh - ln, lg - lh - ln )), b=-sgns ) - log_denom
            return lognegobj

            
    def _hellsq_est(self, i):
        samples = self._sample_g(i, self.n_samples)
        lf = self.target(samples)
        lg = self._logg(samples, i)
        ln = np.log(self.n_samples)
        return 1. - np.exp(logsumexp(lf-lg-ln) - 0.5*logsumexp(2*lf-2*lg-ln))

    def _logfg_est(self, i):
        #estimates log<f, g_i> using samples from g_i^2
        samples = self.D.sample(self.params[i], self.n_samples)
        lf = self.target(samples)
        lg = 0.5 * self.D.logpdf(self.params[i], samples)
        ln = np.log(self.n_samples)
        return logsumexp(lf - lg - ln)
    
    def _logg(self, samples, i):
        #returns the log of g = sum_i lmb_i * g_i, g_i = N(mu_i, Sig_i)^{1/2}
        if i > 0:
            logg_x = 0.5 * self.D.logpdf(self.params[:i+1], samples)
            return logsumexp(logg_x + np.log(np.maximum(self.g_w[:i+1], 1e-64)), axis=1)
        else:
            return -np.inf*np.ones(samples.shape[0])
        
    def _sample_g(self, i, n):
        #samples from g^2
        g_samples = np.zeros((n, self.D.d))
        #compute # samples in each mixture pair
        g_ps = (self.g_w[:i+1, np.newaxis]*self.Z[:i+1, :i+1]*self.g_w[:i+1]).flatten()
        g_ps /= g_ps.sum()
        pair_samples = np.random.multinomial(n, g_ps).reshape((i+1, i+1))
        #symmetrize (will just use lower triangular below for efficiency)
        pair_samples = pair_samples + pair_samples.T
        for k in range(i+1):
            pair_samples[k,k] /= 2
        #invert sigs
        #fill in samples
        cur_idx = 0
        for j in range(i+1):
            for m in range(j+1):
                n_samps = pair_samples[j,m]
                g_samples[cur_idx:cur_idx+n_samps, :] = self.D.cross_sample(self.params[j], self.params[m], n_samps)
                cur_idx += n_samps
        return g_samples
       
       
        