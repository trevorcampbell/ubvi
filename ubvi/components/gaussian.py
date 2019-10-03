import autograd.numpy as np
from component import Component

class Gaussian(Component):
    
    def __init__(self, d, diag):
        super().__init__(d)
        self.diag = diag
        if self.diag:
            self.dim = self.d + self.d
        else:
            self.dim = self.d + self.d**2
        
    def reparam(self, Params):
        Params = np.atleast_2d(Params)
        N = Params.shape[0]
        g_mu = Params[:,:self.d]
        if self.diag:
            g_lSig = Params[:, self.d:]
            return {"g_mu": g_mu, "g_Sig": np.exp(g_lSig)}
        else:
            L = Params[:, self.d:].reshape((N, self.d, self.d))
            g_Sig = np.array([np.dot(l, l.T) for l in L])
            g_Siginv = np.array([np.linalg.inv(sig) for sig in g_Sig])
            return {"g_mu": g_mu, "g_Sig": g_Sig, "g_Siginv": g_Siginv}
    
    def logpdf(self, Params, X):
        Theta = self.reparam(Params)
        if len(X.shape)==1 and self.d==1:
            # each row an observation
            X = X[:,np.newaxis]
        X = np.atleast_2d(X)
        Mu = Theta['g_mu']
        Sig = Theta['g_Sig']
        if self.diag:
            logp = -0.5*Mu.shape[1]*np.log(2*np.pi) - 0.5*np.sum(np.log(Sig), axis=1) - 0.5*np.sum((X[:,np.newaxis,:]-Mu)**2/Sig, axis=2)
        else:
            Siginv = Theta['g_Siginv']
            logp = -0.5*Mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((X[:,np.newaxis,:]-Mu)*((Siginv*((X[:,np.newaxis,:]-Mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)
        if logp.shape[1]==1:
            logp = logp[:,0]
        return logp
            
    def sample(self, params, n):
        std_samples = np.random.randn(n, self.d)
        mu = params[:self.d]
        if self.diag:
            lsig = params[self.d:]
            sd = np.exp(lsig/2)
            return mu + std_samples*sd
        else:
            L = params[self.d:].reshape((self.d, self.d))
            return mu + np.dot(std_samples, L)
    
    def cross_sample(self, params1, params2, n_samps):
        Theta = self.reparam(np.vstack((params1, params2)))
        Mu = Theta['g_mu']
        Sig = Theta['g_Sig']
        if self.diag:
            Sigp = 2./ (1/Sig[0] + 1/Sig[1])
            mup = 0.5*Sigp*(Mu[0]/Sig[0] + Mu[1]/Sig[1])
            return mup + np.sqrt(Sigp)*np.random.randn(n_samps, self.d)
        else:
            Siginv = Theta['g_Siginv']
            Sigp = 2.0*np.linalg.inv(Siginv[0]+Siginv[1])
            mup = 0.5*np.dot(Sigp, np.dot(Siginv[0], Mu[0]) + np.dot(Siginv[1], Mu[1]))
            return np.random.multivariate_normal(mup, Sigp, n_samps)
    
    def log_sqrt_pair_integral(self, params, Prms):
        mu = params[:self.d]
        Prms = np.atleast_2d(Prms)
        Mu = Prms[:, :self.d]
        if self.diag:
            lsig = params[self.d:]
            Lsig = Prms[:, self.d:]
            lSig2 = np.log(0.5)+np.logaddexp(lsig, Lsig)
            return -0.125*np.sum(np.exp(-lSig2)*(mu- Mu)**2, axis=1) - 0.5*np.sum(lSig2, axis=1) + 0.25*np.sum(lsig) + 0.25*np.sum(Lsig, axis=1)
        else:
            l = params[self.d:].reshape((self.d, self.d))
            sig = np.dot(l, l.T)
            N = Prms.shape[0]
            Ls = Prms[:, self.d:].reshape((N, self.d, self.d))
            Sig = np.array([np.dot(L, L.T) for L in Ls])
            Sig2 = 0.5*(sig + Sig)
            return -0.125*((mu - Mu) * np.linalg.solve(Sig2, mu - Mu)).sum(axis=1) - 0.5*np.linalg.slogdet(Sig2)[1] + 0.25*np.linalg.slogdet(sig)[1] + 0.25*np.linalg.slogdet(Sig)[1]
    
    def params_init(self, params, w, inflat):
        params = np.atleast_2d(params)
        i = params.shape[0]
        if i==0:
            mu0 = np.random.multivariate_normal(np.zeros(self.d), inflat*np.eye(self.d))
            if self.diag:
                lSig = np.zeros(self.d)
                xtmp = np.hstack((mu0, lSig))
            else:
                L0 = np.eye(self.d)
                xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
        else:
            mu = params[:, :self.d]
            k = np.random.choice(np.arange(i), p=(w**2)/(w**2).sum())
            if self.diag:
                lsig = params[:, self.d:]
                mu0 = mu[k,:] + np.random.randn(self.d)*np.sqrt(inflat)*np.exp(lsig[k,:])
                LSig = np.random.randn(self.d) + lsig[k,:] 
                xtmp = np.hstack((mu0, LSig))
            else:
                Ls = params[:, self.d:].reshape((i, self.d, self.d))
                sig = np.array([np.dot(L, L.T) for L in Ls])
                mu0 = np.random.multivariate_normal(mu[k,:], inflat*sig[k,:,:])
                L0 = np.exp(np.random.randn())*sig[k,:,:]
                xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
        return xtmp
    
    def print_perf(self, x, itr, gradient, print_every, obj):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'Mu', 'Log(Sig)' if self.diag else 'Eigvals(Sig)', 'GradNorm', 'Boosting Obj'))
        if itr % print_every == 0:
            if self.diag:
                print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(self.d,4)]), str(x[self.d:self.d+min(self.d,4)]), np.sqrt((gradient**2).sum()), obj(x, itr)))
            else:
                L = x[self.d:].reshape((self.d,self.d))
                print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(self.d,4)]), str(np.linalg.eigvalsh(np.dot(L,L.T))[:min(self.d,4)]), np.sqrt((gradient**2).sum()), obj(x, itr)))
            
            
