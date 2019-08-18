import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad

from boostingvi import GradientBoostingVI



class BBVI(GradientBoostingVI):
    def __init__(self, logp, N, d, diag, n_samples, lmb, adam_learning_rate=lambda itr: 1./(itr+1.), adam_num_iters=1000, n_init=1, init_inflation=16,
                 print_every=100):
        super().__init__(logp, N, d, diag, n_samples, n_init, init_inflation, adam_learning_rate, adam_num_iters, print_every)
        self.lmb = lmb
    
    def _new_weights(self, i):
        if i==0:
            return 1
        obj = lambda z, itr : self._kl_estimate(self.target, np.atleast_2d(self.g_mu[:i+1,:]), np.atleast_2d(self.g_Sig[:i+1,:,:]), np.atleast_2d(self.g_Siginv[:i+1,:,:]), z, self.n_samples, self.d)
        grd = grad(obj)
        x = np.ones(np.atleast_2d(self.g_Sig[:i+1,:,:]).shape[0])/float(np.atleast_2d(self.g_Sig[:i+1,:,:]).shape[0])
        return self._simplex_sgd(grd, x, step_size=0.1, num_iters=self.adam_num_iters, callback=lambda prms, itr, grd : self._print_perf_w(prms, itr, grd, self.print_every, self.d, obj))
        
    def _current_distance(self, i):
        print('Optimal mean: ' + str(self.g_mu[i,:]))
        print('Optimal cov: ' + str(self.g_Sig[i,:,:]))
        print('New weights: ' + str(self.g_w[:i+1]))
        #compute current KL estimate
        kl = self._kl_estimate(self.target, self.g_mu[:i+1,:], self.g_Sig[:i+1,:,:], self.g_Siginv[:i+1,:,:], self.g_w[:i+1], self.n_samples, self.d)
        print('New KL estimate: ' + str(kl))
        
    def _objective(self, mu, L, i):
        #compute the covariance matrix from cholesky
        Sig = np.dot(L, np.atleast_2d(L).T)
        #get samples from h
        std_samples = np.random.randn(self.n_samples, self.d)
        h_samples = mu+np.dot(std_samples, np.atleast_2d(L).T)
        #compute log densities
        lf = self.target(h_samples).mean()
        if self.g_w[:i].shape[0] > 0:
            lg = self._mvnlogpdf(h_samples[:,np.newaxis] if len(h_samples.shape)==1 else h_samples, np.atleast_2d(self.g_mu[:i, :]), np.atleast_2d(self.g_Sig[:i, :, :]), np.atleast_2d(self.g_Siginv[:i,:,:]))
            lg = logsumexp(lg+np.log(np.maximum(self.g_w[:i], 1e-64)), axis=1).mean()
        else:
            lg = 0.
        lh = self._mvnlogpdf(h_samples[:, np.newaxis] if len(h_samples.shape)==1 else h_samples, mu, Sig, None).mean()
        return lg + self.lmb(i)*lh - lf
    
    def _kl_estimate(self, logp, g_mu, g_Sig, g_Siginv, g_w, n_samples, d):
        out = 0.
        for k in range(g_w.shape[0]):
            samples = np.random.multivariate_normal(g_mu[k,:], g_Sig[k,:,:], n_samples)
            lg = self._mvnlogpdf(samples, g_mu, g_Sig, g_Siginv)
            lg = logsumexp(lg+np.log(np.maximum(g_w, 1e-64)), axis=1)
            lf = logp(samples)
            out += g_w[k]*(lg.mean()-lf.mean())
        return out
    
    def _simplex_sgd(self, grad, x, callback=None, num_iters=100, step_size=0.001):
        for i in range(num_iters):
            g = grad(x, i)
            #project gradient onto simplex
            g -= g.dot(np.ones(g.shape[0]))*np.ones(g.shape[0])/g.shape[0]
            if callback: callback(x, i, g)
            #compute the step size
            step = step_size/(1.+i) #max(0, i-num_iters/2.))
            #take the step
            x -= step*g
            #account for numerical precision stuff
            x /= x.sum()
            #if left simplex, project
            if np.any(x<0):
                x = self._simplex_projection(x)
        return x
    
    def _simplex_projection(self, x):
        u = np.sort(x)[::-1]
        idcs = np.arange(1, u.shape[0]+1)
        rho_nz = u + 1./idcs*(1.-np.cumsum(u)) > 0
        rho = idcs[rho_nz].max()
        lmb = 1./rho*(1. - u[:rho].sum())
        out = np.maximum(x+lmb, 0.)
        return out/out.sum()

    def _print_perf(self, x, itr, gradient, print_every, d, obj):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'Mu', 'Eigvals(Sig)', 'GradNorm', 'Boosting Obj'))
        if itr % print_every == 0:
            L = x[d:].reshape((d,d))
            print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(d,4)]), str(np.linalg.eigvalsh(np.dot(L,L.T))[:min(d,4)]), np.sqrt((gradient**2).sum()), obj(x, itr)))

    def _print_perf_w(self, x, itr, gradient, print_every, d, obj):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
        if itr % print_every == 0:
            print("{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x), np.sqrt((gradient**2).sum()), obj(x, itr)))

      
        
    
    