import autograd.numpy as np
from autograd import grad
import time



class BoostingVI(object):
    
    def __init__(self, target, N, distribution):
        self.target = target
        self.N = N
        self.D = distribution
        self.g_w = np.zeros(N)
        self.G_w = np.zeros((N,N))
        self.cput = np.zeros(N)                                                                                                                                                                                                                                                                                                         
        self.params = np.zeros((N, self.D.dim))
        self.components = None
        
    def _objective(self):
        raise NotImplementedError
        
    def _new_component(self, i):
        raise NotImplementedError
        
    def _new_weights(self, i):
        raise NotImplementedError
    
    def build(self):
        raise NotImplementedError
        


class GradientBoostingVI(BoostingVI):
    
    def __init__(self, target, N, distribution, n_samples, n_init, adam_learning_rate, adam_num_iters, print_every):
        super().__init__(target, N, distribution)
        self.n_samples = n_samples
        self.n_init = n_init
        self.adam_learning_rate = adam_learning_rate
        self.adam_num_iters = adam_num_iters
        self.print_every = print_every
        self.init_inflation = 1
        
    def build(self):
        for i in range(self.N):
            t0 = time.process_time()
            optimized_params = self._new_component(i)
            self.params[i] = optimized_params
            self.g_w[:i+1] = self._new_weights(i)
            self.G_w[i,:i+1] = self.g_w[:i+1]
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
        opt_params = self._adam(grd, x0)
        print("Comoponent optimization complete.")
        return opt_params
    
    def _initialize(self, obj, i):
        print("Initializing ... ")
        x0 = None
        obj0 = np.inf
        for n in range(self.n_init):
            xtmp = self.D.params_init(self.params[:i, :], self.g_w[:i], self.init_inflation)
            objtmp = obj(xtmp)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                print('improved x0: ' + str(x0) + ' with obj0 = ' + str(obj0))
        if x0 is None:
            raise ValueError
        else:
            return x0
    
    def _adam(self, grad, x):
        b1=0.9
        b2=0.999
        eps=10**-8
        callback=None
        m = np.zeros(len(x))
        v = np.zeros(len(x))
        for i in range(self.adam_num_iters):
            g = grad(x)
            if callback: callback(x, i, g)
            m = (1 - b1) * g      + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1**(i + 1))    # Bias correction.
            vhat = v / (1 - b2**(i + 1))
            x = x - self.adam_learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
        return x
    
    def _current_distance(self):
        raise NotImplementedError

        
        