import autograd.numpy as np
from .component import Component

class Gaussian(Component):
    
    def __init__(self, d, diag): #d is dimension of the space, diag is whether to use diagonal covariance or full
        self.d = d 
        self.diag = diag
        
    def unflatten(self, params):
        params = np.atleast_2d(params)
        N = params.shape[0]
        mu = params[:,:self.d]
        if self.diag:
            logSig = params[:, self.d:]
            return {"mus": mu, "Sigs": np.exp(logSig)}
        else:
            L = params[:, self.d:].reshape((N, self.d, self.d))
            Sig = np.array([np.dot(l, l.T) for l in L])
            Siginv = np.array([np.linalg.inv(sig) for sig in Sig])
            return {"mus": mu, "Sigs": Sig, "Siginvs": Siginv}

    def logpdf(self, params, X):
        theta = self.unflatten(params)
        if len(X.shape)==1 and self.d==1:
            # need to add a dimension so that each row is an observation
            X = X[:,np.newaxis]
        X = np.atleast_2d(X)
        mu = theta['mus']
        Sig = theta['Sigs']
        if self.diag:
            logp = -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.sum(np.log(Sig), axis=1) - 0.5*np.sum((X[:,np.newaxis,:]-mu)**2/Sig, axis=2)
        else:
            Siginv = theta['Siginvs']
            logp = -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((X[:,np.newaxis,:]-mu)*((Siginv*((X[:,np.newaxis,:]-mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)
        if logp.shape[1]==1:
            logp = logp[:,0]
        return logp
            
    def sample(self, param, n):
        std_samples = np.random.randn(n, self.d)
        mu = param[:self.d]
        if self.diag:
            lsig = param[self.d:]
            sd = np.exp(lsig/2)
            return mu + std_samples*sd
        else:
            L = param[self.d:].reshape((self.d, self.d))
            return mu + np.dot(std_samples, L)
    
    def cross_sample(self, param1, param2, n_samps):
        theta = self.unflatten(np.vstack((param1, param2)))
        mu = theta['mus']
        Sig = theta['Sigs']
        if self.diag:
            Sigp = 2./ (1/Sig[0, :] + 1/Sig[1, :])
            mup = 0.5*Sigp*(mu[0, :]/Sig[0, :] + mu[1, :]/Sig[1, :])
            return mup + np.sqrt(Sigp)*np.random.randn(n_samps, self.d)
        else:
            Siginv = theta['Siginvs']
            Sigp = 2.0*np.linalg.inv(Siginv[0, :, :]+Siginv[1, :, :])
            mup = 0.5*np.dot(Sigp, np.dot(Siginv[0,:,:], mu[0,:]) + np.dot(Siginv[1,:,:], mu[1,:]))
            return np.random.multivariate_normal(mup, Sigp, n_samps)

    def log_sqrt_pair_integral(self, new_param, old_params):
        old_params = np.atleast_2d(old_params)
        mu_new = new_param[:self.d]
        mus_old = old_params[:, :self.d]
        if self.diag:
            lsig_new = params[self.d:]
            lsigs_old = old_params[:, self.d:]
            lSig2 = np.log(0.5)+np.logaddexp(lsig_new, lsig_old)
            return -0.125*np.sum(np.exp(-lSig2)*(mu_new - mus_old)**2, axis=1) - 0.5*np.sum(lSig2, axis=1) + 0.25*np.sum(lsig_new) + 0.25*np.sum(lsigs_old, axis=1)
        else:
            L_new = new_param[self.d:].reshape((self.d, self.d))
            Sig_new = np.dot(L_new, L_new.T)
            N = old_params.shape[0]
            Ls_old = old_params[:, self.d:].reshape((N, self.d, self.d))
            Sigs_old = np.array([np.dot(L, L.T) for L in Ls_old])
            Sig2 = 0.5*(Sig_new + Sigs_old)
            return -0.125*((mu_new - mus_old) * np.linalg.solve(Sig2, mu_new - mus_old)).sum(axis=1) - 0.5*np.linalg.slogdet(Sig2)[1] + 0.25*np.linalg.slogdet(Sig_new)[1] + 0.25*np.linalg.slogdet(Sigs_old)[1]
    
    def params_init(self, params, weights, inflation):
        params = np.atleast_2d(params)
        i = params.shape[0]
        if i==0:
            mu0 = np.random.multivariate_normal(np.zeros(self.d), inflation*np.eye(self.d))
            if self.diag:
                lSig = np.zeros(self.d)
                xtmp = np.hstack((mu0, lSig))
            else:
                L0 = np.eye(self.d)
                xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
        else:
            mu = params[:, :self.d]
            k = np.random.choice(np.arange(i), p=(weights[-1]**2)/(weights[-1]**2).sum())
            if self.diag:
                lsig = params[:, self.d:]
                mu0 = mu[k,:] + np.random.randn(self.d)*np.sqrt(inflation)*np.exp(lsig[k,:])
                LSig = np.random.randn(self.d) + lsig[k,:] 
                xtmp = np.hstack((mu0, LSig))
            else:
                Ls = params[:, self.d:].reshape((i, self.d, self.d))
                sig = np.array([np.dot(L, L.T) for L in Ls])
                mu0 = np.random.multivariate_normal(mu[k,:], inflation*sig[k,:,:])
                L0 = np.exp(np.random.randn())*sig[k,:,:]
                xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
        return xtmp
    
    def print_perf(self, itr, x, obj, grd):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'mu', 'Log(Sig)' if self.diag else 'Eigvals(Sig)', 'GradNorm', 'Boosting Obj'))
        if self.diag:
            print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(self.d,4)]), str(x[self.d:self.d+min(self.d,4)]), np.sqrt((grd**2).sum()), obj))
        else:
            L = x[self.d:].reshape((self.d,self.d))
            print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(self.d,4)]), str(np.linalg.eigvalsh(np.dot(L,L.T))[:min(self.d,4)]), np.sqrt((grd**2).sum()), obj))
            
            
