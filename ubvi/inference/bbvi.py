import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad

from .boostingvi import BoostingVI
from ..optimization import simplex_sgd



class BBVI(BoostingVI):
    
    def __init__(self, logp, N, distribution, optimization, n_samples, n_init, lmb, n_simplex_iters = 3000):
        super().__init__(logp, N, distribution, optimization)
        self.lmb = lmb
        self.n_samples = n_samples
        self.n_init = n_init
        self.init_inflation = 100
        self.n_simplex_iters = n_simplex_iters
    
    
    def _weights_update(self):
        i = self.params.shape[0]
        assert i > self.g_w.shape[0]
        if i==1:
            return 1
        else:
            obj = lambda z: self._kl_estimate(self.params, z)
            grd = grad(obj)
            x = np.ones(i)/float(i)
            return simplex_sgd(x, grd, learning_rate=lambda itr : 0.1/(1+itr), num_iters=self.n_simplex_iters, callback=lambda x, itr, gradient: self.print_perf_w(x, itr, gradient, obj))
    
    
    def _error(self):
        print('New weights: ' + str(self.g_w))
        i = self.params.shape[0]
        assert self.g_w.shape[0] == i and i > 0
        kl = self._kl_estimate(self.params, self.g_w)
        print('New KL estimate: ' + str(kl))
        
    
    def _objective(self, x, itr):
        assert self.g_w.shape[0] == self.params.shape[0]
        h_samples = self.D.sample(x, self.n_samples)
        #compute log densities
        lf = self.target(h_samples).mean()
        i = self.g_w.shape[0]
        if i > 0:
            lg = self.D.logpdf(self.params, h_samples)
            if i==1:
                lg = lg[:,np.newaxis]
            lg = logsumexp(lg+np.log(np.maximum(self.g_w, 1e-64)), axis=1).mean()
        else:
            lg = 0.
        lh = self.D.logpdf(x, h_samples).mean()
        return lg + self.lmb(i)*lh - lf
    
    
    def _kl_estimate(self, Params, W):
        out = 0.
        for k in range(W.shape[0]):
            samples = self.D.sample(Params[k], self.n_samples)
            lg = self.D.logpdf(Params, samples)
            if len(lg.shape)==1:
                lg = lg[:,np.newaxis]
            lg = logsumexp(lg+np.log(np.maximum(W, 1e-64)), axis=1)
            lf = self.target(samples)
            out += W[k]*(lg.mean()-lf.mean())
        return out
    
    def print_perf_w(self, x, itr, gradient, obj):
        if itr == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
        if itr % self.print_every == 0:
            print("{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x), np.sqrt((gradient**2).sum()), obj(x)))

      
    
