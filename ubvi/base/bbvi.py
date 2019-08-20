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
        else:
            return self._bbvi_new_weights(self.target, self.g_mu[:i+1], self._g_Sig[:i+1], self.g_Siginv[:i+1], self.n_samples, self.print_every, self.d, self.diag, self.adam_num_iters, self.n_init)
        
    def _bbvi_new_weights(self, logp, g_mu, g_Sig, g_Siginv, n_samples, print_every, d, diag, num_opt_itrs, n_init=10):
        obj = lambda z, itr : self._kl_estimate(logp, np.atleast_2d(g_mu), np.atleast_2d(g_Sig), np.atleast_2d(g_Siginv), z, n_samples, d, diag)
        grd = grad(obj)
        x = np.ones(g_Sig.shape[0])/float(g_Sig.shape[0])
        return self._simplex_sgd(grd, x, step_size=0.1, num_iters=num_opt_itrs, callback=lambda prms, itr, grd : self._print_perf_w(prms, itr, grd, print_every, d, obj))
        
    def _current_distance(self, i):
        print('Optimal mean: ' + str(self.g_mu[i]))
        print('Optimal cov: ' + str(self.g_Sig[i]))
        print('New weights: ' + str(self.g_w[:i+1]))
        #compute current KL estimate
        kl = self._kl_estimate(self.target, self.g_mu[:i+1], self.g_Sig[:i+1], self.g_Siginv[:i+1], self.g_w[:i+1], self.n_samples, self.d, self.diag)
        print('New KL estimate: ' + str(kl))
        
    def _objective(self, logp, mu, L, g_mu, g_Sig, g_Siginv, g_w, n_samples, diag):
        return self._bbvi_obj(logp, mu, L, g_mu, g_Sig, g_Siginv, g_w, self.lmb, n_samples, diag)
    
    def _bbvi_obj(self, logp, mu, L, g_mu, g_Sig, g_Siginv, g_w, lmb, n_samples, diag):
        d = g_mu.shape[1]
        #compute the covariance matrix from cholesky
        Sig = L if diag else np.dot(L, np.atleast_2d(L).T)
        #get samples from h
        std_samples = np.random.randn(n_samples, d)
        h_samples = mu+np.exp(0.5*Sig)*std_samples if diag else mu+np.dot(std_samples, np.atleast_2d(L).T)
        #compute log densities
        lf = logp(h_samples).mean()
        if g_w.shape[0] > 0:
            lg = self._mvnlogpdf(h_samples[:,np.newaxis] if len(h_samples.shape)==1 else h_samples, g_mu, g_Sig, g_Siginv, diag)
            lg = logsumexp(lg+np.log(np.maximum(g_w, 1e-64)), axis=1).mean()
        else:
            lg = 0.
        #lh = stats.multivariate_normal.logpdf(h_samples, mu, Sig).mean()
        lh = self.mvnlogpdf(h_samples[:, np.newaxis] if len(h_samples.shape)==1 else h_samples, mu, Sig, None, diag).mean()
        #lh_exact = 0.5*np.linalg.slogdet(2.*np.pi*np.exp(1)*Sig)[1]
        return lg+lmb*lh-lf
    
    def _kl_estimate(self, logp, g_mu, g_Sig, g_Siginv, g_w, n_samples, d, diag):
        out = 0.
        for k in range(g_w.shape[0]):
            samples = g_mu[k]+np.random.randn(n_samples, d)*np.exp(0.5*g_Sig[k]) if diag else np.random.multivariate_normal(g_mu[k], g_Sig[k], n_samples)
            lg = self._mvnlogpdf(samples, g_mu, g_Sig, g_Siginv, diag)
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

    def _print_perf(self, x, itr, gradient, print_every, d, obj, diag):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'Mu', 'Log(Sig)' if diag else 'Eigvals(Sig)', 'GradNorm', 'Boosting Obj'))
        if itr % print_every == 0:
            L = x[d:].reshape((d,d))
            print("{:^30}|{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x[:min(d,4)]), str(x[d:d+min(d,4)]) if diag else str(np.linalg.eigvalsh(np.dot(L,L.T))[:min(d,4)]), np.sqrt((gradient**2).sum()), obj(x, itr)))

    def _print_perf_w(self, x, itr, gradient, print_every, d, obj):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
        if itr % print_every == 0:
            print("{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x), np.sqrt((gradient**2).sum()), obj(x, itr)))

      
        
    
    