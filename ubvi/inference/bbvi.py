import autograd.numpy as np
from ..autograd import logsumexp
from autograd import grad

from .boostingvi import BoostingVI
from ..optimization import simplex_sgd



class BBVI(BoostingVI):
    
    def __init__(self, logp, component_dist, opt_alg, lmb = lambda itr : 1, n_samples = 100, n_simplex_iters = 3000, **kw):
        super().__init__(component_dist, opt_alg, **kw)
        self.logp = logp
        self.lmb = lmb
        self.n_samples = n_samples
        self.n_simplex_iters = n_simplex_iters
    
    def _compute_weights(self):
        if self.params.shape[0] == 1:
            return 1
        else:
            obj = lambda z, i: self._kl_estimate(self.params, z)
            grd = grad(obj)
            x = np.ones(self.params.shape[0])/float(self.params.shape[0])
            return simplex_sgd(x, obj, grd, learning_rate=lambda itr : 0.1/(1+itr), num_iters=self.n_simplex_iters, callback = self._print_perf_w if self.verbose else None)

    def _error(self):
        return "KL Divergence", self._kl_estimate(self.params, self.weights[-1])
    
    def _objective(self, x, itr):
        h_samples = self.component_dist.sample(x, self.n_samples)
        #compute log target density under samples
        lf = self.logp(h_samples).mean()
        #compute current log mixture density
        if len(self.weights) > 0:
            lg = self.component_dist.logpdf(self.params, h_samples)
            if len(lg.shape) == 1:
                #need to add a dimension so that each sample corresponds to a row in lg
                lg = lg[:,np.newaxis] 
            lg = logsumexp(lg+np.log(np.maximum(self.weights[-1], 1e-64)), axis=1).mean()
        else:
            lg = 0.
        lh = self.component_dist.logpdf(x, h_samples).mean()
        return lg + self.lmb(itr)*lh - lf
    
    def _kl_estimate(self, prms, wts):
        out = 0.
        for k in range(wts.shape[0]):
            samples = self.component_dist.sample(prms[k, :], self.n_samples)
            lg = self.component_dist.logpdf(prms, samples)
            if len(lg.shape)==1:
                lg = lg[:,np.newaxis]
            lg = logsumexp(lg+np.log(np.maximum(wts, 1e-64)), axis=1)
            lf = self.logp(samples)
            out += wts[k]*(lg.mean()-lf.mean())
        return out
    
    def _print_perf_w(self, itr, x, obj, grd, print_every = 100):
        if itr % (10*print_every) == 0:
            print("{:^30}|{:^30}|{:^30}|{:^30}".format('Iteration', 'W', 'GradNorm', 'KL'))
        if itr % print_every == 0:
            print("{:^30}|{:^30}|{:^30.2f}|{:^30.2f}".format(itr, str(x), np.sqrt((grd**2).sum()), obj))

      
    
