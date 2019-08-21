import autograd.numpy as np
from scipy.optimize import nnls
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats

from boostingvi import GradientBoostingVI



class UBVI(GradientBoostingVI):
    
    def __init__(self, logf, N, d, diag, n_samples, n_logfg_samples, adam_learning_rate=lambda itr: 1./(1.+itr), init_inflation=16, n_init=1, adam_num_iters=1000, 
                 print_every=100):
        super().__init__(logf, N, d, diag, n_samples, n_init, init_inflation, adam_learning_rate, adam_num_iters, print_every)
        self.n_logfg_samples = n_logfg_samples
        self.Z = np.zeros((self.N, self.N))
        self._logfg = -np.inf*np.ones(self.N)
        self._logfgsum = -np.inf
        
    def _update(self, i, optimized_params):
        super()._update(i, optimized_params)
        print('Updating Z...')
        Znew = np.exp(self._log_sqrt_pair_integral(self.g_mu[i], self._g_Sig[i], self.g_mu[:i+1], self._g_Sig[:i+1]))
        self.Z[i, :i+1] = Znew
        self.Z[:i+1, i] = Znew
        print('Updating log<f,g>...')
        #add a new logfg
        self._logfg[i] = self._logfg_est(self.target, self.g_mu[i], self._g_Sig[i], self.n_logfg_samples)
        
    def _new_weights(self, i):
        if i==0:
            return 1
        else:
            return self._ubvi_new_weights(self.Z[:i+1,:i+1], self.logfg[:i+1])
        
    def _ubvi_new_weights(self, Z, logfg):
        Linv = np.linalg.inv(np.linalg.cholesky(Z))
        d = np.exp(logfg-logfg.max()) #the opt is invariant to d scale, so normalize to have max 1
        b = nnls(Linv, -np.dot(Linv, d))[0]
        lbd = np.dot(Linv, b+d)
        return np.maximum(0., np.dot(Linv.T, lbd/np.sqrt(((lbd**2).sum()))))
        
    def _current_distance(self, i):
        #compute current log<f, g> estimate
        self._logfgsum = logsumexp(np.hstack((-np.inf, self._logfg[:i+1] + np.log(np.maximum(self.g_w[:i+1], 1e-64)))))
        print('Optimal mean: ' + str(self.g_mu[i]))
        print('Optimal cov: ' + str(self.g_Sig[i]))
        print('New weights: ' + str(self.g_w[:i+1]))
        print('New Z: ' + str(self.Z[:i+1,:i+1]))
        print('New log<f,g>: ' + str(self._logfg[:i+1]))
        hellsq = self._hellsq_est(self.target, self.g_mu[:i+1], self.g_Sig[:i+1], self.g_Siginv[:i+1], self.g_w[:i+1], self.Z[:i+1,:i+1], self.n_logfg_samples, self.diag)
        print('New Hellinger-Squared estimate: ' + str(hellsq))
     
    def _objective(self, logf, mu, L, g_mu, g_Sig, g_Siginv, g_lmb, n_samples, diag):
        return self._ubvi_obj(logf, mu, L, g_mu, g_Sig, g_Siginv, g_lmb, n_samples, diag, self._logfgsum, allow_negative=False)
    
    def _ubvi_obj(self, logf, mu, L, g_mu, g_Sig, g_Siginv, g_lmb, n_samples, diag, logfg, allow_negative=False):
        # L represents the cholesky decomp of covariance when using general cov structure;
        # L represents the logrithm of diagnal variance when using diagnal structure cov
        d = g_mu.shape[1]
        #compute the covariance matrix from cholesky
        Sig = L if diag else np.dot(L, np.atleast_2d(L).T)
        lgh = -np.inf if g_lmb.shape[0] == 0 else logsumexp(np.log(np.maximum(g_lmb, 1e-64)) + self._log_sqrt_pair_integral(mu, Sig, g_mu, g_Sig, diag))
        #get samples from h
        std_samples = np.random.randn(n_samples, d)
        h_samples = mu+np.exp(0.5*L)*std_samples if diag else mu+np.dot(std_samples, np.atleast_2d(L).T)
        lf = logf(h_samples)
        lh = 0.5*self._mvnlogpdf(h_samples, mu, Sig, None, diag) #stats.multivariate_normal.logpdf(h_samples, mu, Sig)
        ln = np.log(n_samples)
        lf_num = logsumexp(lf - lh - ln)
        lg_num = logfg + lgh 
        log_denom = 0.5*np.log(1.-np.exp(2*lgh))
        if lf_num > lg_num:
            logobj = lf_num - log_denom + np.log(1.-np.exp(lg_num-lf_num))
            return logobj
        else:
            logobj = lg_num - log_denom + np.log(1.-np.exp(lf_num-lg_num))
            return -logobj
        
    def _log_sqrt_pair_integral(self, mu, Sig, mui, Sigi, diag):
        #returns array of [log < N(mu, Sig)^{1/2},  N(mu_i, Sig_i)^{1/2} >] over i
        if diag:
            lSig2 = np.log(0.5)+np.logaddexp(Sig,Sigi)
            return -0.125*np.sum(np.exp(-lSig2)*(mu-mui)**2, axis=1) - 0.5*np.sum(lSig2, axis=1) + 0.25*np.sum(Sig) + 0.25*np.sum(Sigi, axis=1)
        else:
            Sig2 = 0.5*(Sig+Sigi)
            return -0.125*((mu-mui) * np.linalg.solve(Sig2, mu-mui)).sum(axis=1) - 0.5*np.linalg.slogdet(Sig2)[1] + 0.25*np.linalg.slogdet(Sig)[1] + 0.25*np.linalg.slogdet(Sigi)[1]
    
    def _hellsq_est(self, logf, g_mu, g_Sig, g_Siginv, g_lmb, Z, n_samples, diag):
        samples = self._sample_g(g_mu, g_Sig, g_Siginv, g_lmb, Z, n_samples, diag)
        lf = logf(samples)
        lg = self._logg(samples, g_mu, g_Sig, g_Siginv, g_lmb, diag)
        ln = np.log(n_samples)
        return 1. - np.exp(logsumexp(lf-lg-ln) - 0.5*logsumexp(2*lf-2*lg-ln))

    def _logfg_est(self, logf, mu, Sig, n_samples, Diag):
        #estimates log<f, g_i> using samples from g_i^2
        if Diag:
            Sig = np.diag(np.exp(Sig))
        samples = np.random.multivariate_normal(mu, Sig, n_samples)
        lf = logf(samples)
        lg = 0.5*stats.multivariate_normal.logpdf(samples, mu, Sig)
        ln = np.log(n_samples)
        return logsumexp(lf - lg - ln)
    
    def _logg(self, x, g_mu, g_Sig, g_Siginv, g_lmb, diag):
        #returns the log of g = sum_i lmb_i * g_i, g_i = N(mu_i, Sig_i)^{1/2}
        if g_lmb.shape[0] > 0:
            logg_x = 0.5*self._mvnlogpdf(x[:,np.newaxis] if len(x.shape)==1 else x, g_mu, g_Sig, g_Siginv, diag)
            return logsumexp(logg_x + np.log(np.maximum(g_lmb, 1e-64)), axis=1)
        else:
            return -np.inf*np.ones(x.shape[0])
        
    def _sample_g(self, g_mu, g_Sig, g_Siginv, g_lmb, Z, n_samples, diag):
        #samples from g^2
        g_samples = np.zeros((n_samples, g_mu.shape[1]))
        #compute # samples in each mixture pair
        g_ps = (g_lmb[:, np.newaxis]*Z*g_lmb).flatten()
        g_ps /= g_ps.sum()
        pair_samples = np.random.multinomial(n_samples, g_ps).reshape((g_lmb.shape[0], g_lmb.shape[0]))
        #symmetrize (will just use lower triangular below for efficiency)
        pair_samples = pair_samples + pair_samples.T
        for i in range(pair_samples.shape[0]):
            pair_samples[i,i] /= 2
        #invert sigs
        #fill in samples
        cur_idx = 0
        for j in range(g_lmb.shape[0]):
            for k in range(j+1):
                n_samps = pair_samples[j,k]
                if diag:
                    Sigp = 2./np.exp(np.logaddexp(-g_Sig[j], -g_Sig[k]))
                    mup = 0.5*Sigp*(np.exp(-g_Sig[j])*g_mu[j] + np.exp(-g_Sig[k])*g_mu[k])
                    g_samples[cur_idx:cur_idx+n_samps, :] = mup + np.sqrt(Sigp)*np.random.randn(n_samps, mup.shape[0])
                else:
                    Sigp = 2.0*np.linalg.inv(g_Siginv[j]+g_Siginv[k])
                    mup = 0.5*np.dot(Sigp, np.dot(g_Siginv[j], g_mu[j]) + np.dot(g_Siginv[k], g_mu[k]))
                    g_samples[cur_idx:cur_idx+n_samps, :] = np.random.multivariate_normal(mup, Sigp, n_samps)
                cur_idx += n_samps
        return g_samples
        