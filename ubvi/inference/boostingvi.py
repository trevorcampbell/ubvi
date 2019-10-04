import autograd.numpy as np
from autograd import grad
import time


class BoostingVI(object):
    
    def __init__(self, target, component_dist, opt_alg, n_init = 10, init_inflation = 100, print_error = True):
        self.target = target #the target log density
        self.N = 0 #current # of components
        self.component_dist = component_dist #component distribution object
        self.opt_alg = opt_alg #optimization algorithm function
        self.weights = [] #trace of weights
        self.params = np.empty((0, self.component_dist.dim)) #??
        self.components = None #??
        self.cputs = [] #list of computation times for each build step
        self.n_init = n_init #number of times to initialize each component
        self.init_inflation = init_inflation #number of times to initialize each component
        
    def build(self, N):
	#build the approximation up to N components
        for i in range(self.N, N):
            t0 = time.process_time()
	    #get a new component
            new_param = self._build_new_component()
            #add it to the matrix of flattened parameters
            self.params = np.vstack((self.params, new_param))
            #add the weights
            self.weights.append(np.atleast_1d(self._compute_weights()))
            #print out the current error
            if self.print_error:
                print('Current: ' + self._current_distance())
            #compute the time taken for this step
            self.cputs.append(time.process_time() - t0)
        #??
        self.components = self.component_dist.reparam(self.params)
        #??
        output = self.components
        output.update([('g_w', self.g_w), ('G_w', self.G_w), ('cput', self.cput)])
	#update self.N to the new # comps
        self.N = N
        return output
        
    def _build_new_component(self):
        n = self.params.shape[0]+1
        obj = lambda x, itr: self._objective(x, itr)
        print("Initializing component " + str(n) +"... ")
        x0 = self._initialize(obj)
        print("Initialization of component " + str(n)+ " complete, x0 = " + str(x0))
        grd = grad(obj)
        print("Optimizing component " + str(n) +"... ")
        opt_params = self.opt_alg(x0, obj, grd, callback=lambda prms, itr, grd : self.component_dist.print_perf(prms, itr, grd, self.print_every, obj))
        print("Optimization of component " + str(n) + " complete")
        return opt_params
    
    def _initialize(self, obj):
        x0 = None
        obj0 = np.inf
        #try initializing n_init times
        for n in range(self.n_init):
            xtmp = self.component_dist.params_init(self.params, self.weights[-1], self.init_inflation)
            objtmp = obj(xtmp, -1)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                print('Current best initialization -- x0: ' + str(x0) + ' obj0 = ' + str(obj0))
        if x0 is None:
            #if every single initialization had an infinite objective, just raise an error
            raise ValueError
        #return the initialized result
        return x0
        
    def _objective(self, itr):
        raise NotImplementedError
        
    def _compute_weights(self):
        raise NotImplementedError
    
    def _current_distance(self):
        raise NotImplementedError
