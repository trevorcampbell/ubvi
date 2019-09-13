import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad

from boostingvi import GradientBoostingVI



class BBVI(GradientBoostingVI):
    
    def __init__(self, logp, N, distribution, n_samples, n_init, lmb, adam_learning_rate, adam_num_iters, print_every):
        super().__init__(logp, N, distribution, n_samples, n_init, adam_learning_rate, adam_num_iters, print_every)
        self.lmb = lmb
    
    def _new_weights(self, i):
        if i==0:
            return 1
        else:
            obj = lambda z, itr : self._kl_estimate(self.params[:i+1], z)
            grd = grad(obj)
            x = np.ones(i+1)/float(i+1)
            return self._simplex_sgd(grd, x)
        
    def _current_distance(self, i):
        print('New weights: ' + str(self.g_w[:i+1]))
        #compute current KL estimate
        kl = self._kl_estimate(self.params[:i+1], self.g_w[:i+1])
        print('New KL estimate: ' + str(kl))
    
    def _objective(self, x, Params, W):
        h_samples = self.D.sample(x, self.n_samples)
        #compute log densities
        lf = self.target(h_samples).mean()
        i = W.shape[0]
        if i > 0:
            lg = self.D.logpdf(Params, h_samples)
            lg = logsumexp(lg+np.log(np.maximum(W, 1e-64)), axis=1).mean()
        else:
            lg = 0.
        lh = self.D.logpdf(x, h_samples).mean()
        return lg + self.lmb(i)*lh - lf
    
    def _kl_estimate(self, Params, W):
        out = 0.
        for k in range(W.shape[0]):
            samples = self.D.sample(Params[k], self.n_samples)
            lg = self.D.logpdf(Params, samples)
            lg = logsumexp(lg+np.log(np.maximum(W, 1e-64)), axis=1)
            lf = self.target(samples)
            out += W[k]*(lg.mean()-lf.mean())
        return out
    
    def _simplex_sgd(self, grad, x):
        callback = None
        step_size = 0.1
        for i in range(self.adam_num_iters):
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


      
        
    
    
    