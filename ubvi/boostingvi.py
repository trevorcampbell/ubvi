import autograd.numpy as np
from autograd import grad
import time



class BoostingVI(object):
    
    def __init__(self, target, N, distribution, optimization):
        self.target = target
        self.N = N
        self.D = distribution
        self.Opt = optimization
        self.g_w = np.zeros(N)
        self.G_w = []
        self.params = np.zeros((N, self.D.dim))
        self.components = None
        self.cput = np.zeros(N) 
        
    def build(self):
        for i in range(self.N):
            t0 = time.process_time()
            optimized_params = self._new_component(i)
            self.params[i] = optimized_params
            self.g_w[:i+1] = self._new_weights(i)
            self.G_w.append(self.g_w[:i+1])
            self._current_distance(i)
            self.cput[i] = time.process_time() - t0
        self.components = self.D.reparam(self.params)
        output = self.components
        output.update([('g_w', self.g_w), ('G_w', self.G_w), ('cput', self.cput)])
        return output
        
    def _new_component(self, i):
        obj = lambda x: -self._objective(x, self.params[:i], self.g_w[:i])
        x0 = self._initialize(obj, i)
        grd = grad(obj)
        opt_params = self.Opt.optimize(grd, x0)
        print("Comoponent optimization complete.")
        return opt_params
        
    def _objective(self):
        raise NotImplementedError
        
    def _new_weights(self, i):
        raise NotImplementedError
        
    def _initialize(self, i):
        raise NotImplementedError
    
    def _current_distance(self):
        raise NotImplementedError