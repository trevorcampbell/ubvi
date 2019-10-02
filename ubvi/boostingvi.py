import autograd.numpy as np
from autograd import grad
import time


class BoostingVI(object):
    
    def __init__(self, target, N, distribution, optimization):
        self.target = target
        self.N = N
        self.D = distribution
        self.Opt = optimization
        self.g_w = np.empty((0,0))
        self.G_w = []
        self.params = np.empty((0, self.D.dim))
        self.components = None
        self.cput = np.zeros(N) 
        
    def build(self):
        assert self.params.shape[0] == 0, 'The Boosting VI object can only build onece.'
        for i in range(self.N):
            t0 = time.process_time()
            optimized_params = self._new_component()
            self.params = np.vstack((self.params, optimized_params))
            self.g_w = np.atleast_1d(self._weights_update())
            self.G_w.append(self.g_w)
            self._current_distance()
            self.cput[i] = time.process_time() - t0
        self.components = self.D.reparam(self.params)
        output = self.components
        output.update([('g_w', self.g_w), ('G_w', self.G_w), ('cput', self.cput)])
        return output
        
    def _new_component(self):
        obj = lambda x, itr: self._objective(x, itr)
        x0 = self._initialize(obj)
        grd = grad(obj)
        opt_params = self.Opt.optimize(grd, x0, callback=lambda prms, itr, grd : self.D.print_perf(prms, itr, grd, self.Opt.print_every, obj))
        print("Comoponent optimization complete.")
        return opt_params
    
    def _initialize(self, obj):
        print("Initializing ... ")
        x0 = None
        obj0 = np.inf
        for n in range(self.n_init):
            xtmp = self.D.params_init(self.params, self.g_w, self.init_inflation)
            objtmp = obj(xtmp, -1)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                print('improved x0: ' + str(x0) + ' with obj0 = ' + str(obj0))
        if x0 is None:
            raise ValueError
        else:
            return x0
        
    def _objective(self):
        raise NotImplementedError
        
    def _new_weights(self):
        raise NotImplementedError
    
    def _current_distance(self):
        raise NotImplementedError