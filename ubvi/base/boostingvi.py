import autograd.numpy as np
from autograd import grad
import time



class BoostingVI(object):
    
    def __init__(self, target, N, d, diag):
        self.target = target
        self.N = N
        self.d = d
        self.diag = diag
        self.g_mu = np.zeros((self.N, self.d))
        self.g_Sig = np.zeros((self.N, self.d, self.d))
        self.g_Siginv = np.zeros((self.N, self.d, self.d))
        self.g_w = np.zeros(N)
        self.G_w = np.zeros((N,N))
        self.cput = np.zeros(N)
        
    def build(self):
        raise NotImplementedError
    
    def _objective(self):
        raise NotImplementedError
    
    def _new_component(self):
        raise NotImplementedError
    
    def _new_weights(self):
        raise NotImplementedError
        
    



class GradientBoostingVI(BoostingVI):
    
    def __init__(self, target, N, d, diag, n_samples, n_init, init_inflation, adam_learning_rate, adam_num_iters, print_every):
        super().__init__(target, N, d, diag)
        self.n_samples = n_samples
        self.n_init = n_init
        self.init_inflation = init_inflation
        self.adam_learning_rate = adam_learning_rate
        self.adam_num_iters = adam_num_iters
        self.print_every = print_every
        
    def build(self):
        for i in range(self.N):
            t0 = time.process_time()
            optimized_params = self._new_component(i)
            self._update(i, optimized_params)
            self.g_w[:i+1] = self._new_weights(i)
            self.G_w[i,:i+1] = self.g_w[:i+1]
            self._current_distance(i)
            self.cput[i] = time.process_time() - t0
            
    def _new_component(self, i):
        obj = lambda x, itr: -self._objective(x[:self.d], x[self.d:].reshape((self.d,self.d)), i)
        x0 = self._initialize(obj, i)
        grd = grad(obj)
        optimized_params = self._adam(grd, x0, callback=lambda prms, itr, grd: self._print_perf(prms, itr, grd, self.print_every, self.d, obj))
        print('Component optimization complete')
        return optimized_params
    
    def _update(self, i, optimized_params):
        self.g_mu[i, :] = optimized_params[:self.d]
        L = optimized_params[self.d:].reshape((self.d, self.d))
        self.g_Sig[i, :, :] = np.dot(L, L.T)
        self.g_Siginv[i,:,:] = np.linalg.inv(self.g_Sig[i,:,:])
            
    def _adam(self, grad, x, b1=0.9, b2=0.999, eps=10**-8, callback=None):
        m = np.zeros(len(x))
        v = np.zeros(len(x))
        for i in range(self.adam_num_iters):
            g = grad(x, i)
            if callback: callback(x, i, g)
            m = (1 - b1) * g      + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1**(i + 1))    # Bias correction.
            vhat = v / (1 - b2**(i + 1))
            x = x - self.adam_learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
        return x
        
    def _initialize(self, obj, i):
        print('Initialization')
        x0 = None
        obj0 = np.inf
        for n in range(self.n_init):
            if i==0:
                mu0 = np.random.multivariate_normal(np.zeros(self.d), self.init_inflation*np.eye(self.d))
                L0 = np.eye(self.d)
            else:
                k = np.random.choice(np.arange(i), p=(self.g_w[:i]**2)/(self.g_w[:i]**2).sum())
                mu0 = np.random.multivariate_normal(self.g_mu[k,:], self.init_inflation*self.g_Sig[k,:,:])
                L0 = np.exp(np.random.randn())*self.g_Sig[k,:,:]
            xtmp = np.hstack((mu0, L0.reshape(self.d*self.d)))
            objtmp = obj(xtmp, -1)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                print('improved x0: ' + str(x0) + ' with obj0 = ' + str(obj0))
        if x0==None:
            raise ValueError
        else:
            return x0
        
    def _mvnlogpdf(self, x, mu, Sig, Siginv):
        if len(Sig.shape) > 2:
            #(x[:,np.newaxis,:]-mu) is nxkxd; sig=kxdxd 
            return -0.5*mu.shape[1]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x[:,np.newaxis,:]-mu)*((Siginv*((x[:,np.newaxis,:]-mu)[:,:,np.newaxis,:])).sum(axis=3))).sum(axis=2)
        else:
            return -0.5*mu.shape[0]*np.log(2*np.pi) - 0.5*np.linalg.slogdet(Sig)[1] - 0.5*((x-mu)*np.linalg.solve(Sig, (x-mu).T).T).sum(axis=1)
    
    def _current_distance(self):
        raise NotImplementedError
    
    def _print_perf(self, x, itr, gradient, print_every, d, obj):
        raise NotImplementedError
    
