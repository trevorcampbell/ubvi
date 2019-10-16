import autograd.numpy as np
from autograd import grad
import time
from ..optimization.adam import adam


class BoostingVI(object):
    
    def __init__(self, component_dist, opt_alg, n_init = 10, init_inflation = 100, estimate_error = True, verbose = True):
        self.N = 0 #current num of components
        self.component_dist = component_dist #component distribution object
        self.opt_alg = opt_alg #optimization algorithm function
        self.weights = np.empty(0) #weights
        self.params = np.empty((0, 0)) 
        self.cput = 0. #total computation time so far
        self.error = np.inf #error for the current mixture
        self.n_init = n_init #number of times to initialize each component
        self.init_inflation = init_inflation #number of times to initialize each component
        self.verbose = verbose
        self.estimate_error = estimate_error
        
    def build(self, N):
	#build the approximation up to N components
        for i in range(self.N, N):
            t0 = time.perf_counter()

            #initialize the next component
            if self.verbose: print("Initializing component " + str(i+1) +"... ")
            x0 = self._initialize()

            #if this is the first component, set the dimension of self.params
            if self.params.size == 0:
                self.params = np.empty((0, x0.shape[0]))
            if self.verbose: print("Initialization of component " + str(i+1)+ " complete, x0 = " + str(x0))
            
            #build the next component
            if self.verbose: print("Optimizing component " + str(i+1) +"... ")
            grd = grad(self._objective)
            try:
                new_param = self.opt_alg(x0, self._objective, grd)
            except: #bbvi can run into bad degeneracies; if so, just revert to initialization
                new_param = x0
            if self.verbose: print("Optimization of component " + str(i+1) + " complete")

            #add it to the matrix of flattened parameters
            self.params = np.vstack((self.params, new_param))

            #compute the new weights and add to the list
            if self.verbose: print('Updating weights...')
            self.weights_prev = self.weights.copy()
            try:
                self.weights = np.atleast_1d(self._compute_weights())
            except: #bbvi can run into bad degeneracies; if so, just throw out the new component
                self.weights = np.hstack((self.weights_prev, 0.))

            if self.verbose: print('Weight update complete...')

            #compute the time taken for this step
            self.cput += time.perf_counter() - t0

            #estimate current error if desired
            if self.estimate_error:
                err_name, self.error = self._error()

            #print out the current error
            if self.verbose:
                print('Component ' + str(self.params.shape[0]) +':')
                print('Cumulative CPU Time: ' + str(self.cput))
                if self.estimate_error:
                    print(err_name +': ' + str(self.error))
                print('Params:' + str(self.component_dist.unflatten(self.params)))
                print('Weights: ' + str(self.weights))
            
        #update self.N to the new # comps
        self.N = N

        #generate the nicely-formatted output params
        output = self._get_mixture()
        output['cput'] = self.cput
        output['obj'] = self.error
        return output
        
        
    def _initialize(self):
        x0 = None
        obj0 = np.inf
        t0 = time.perf_counter()
        #try initializing n_init times
        for n in range(self.n_init):
            xtmp = self.component_dist.params_init(self.params, self.weights, self.init_inflation)
            objtmp = self._objective(xtmp, -1)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
            if self.verbose and (n == 0 or n == self.n_init - 1 or time.perf_counter() - t0 > 0.5):
                if n == 0:
                    print("{:^30}|{:^30}|{:^30}".format('Iteration', 'Best x0', 'Best obj0'))
                print("{:^30}|{:^30}|{:^30.3f}".format(n, str(x0), obj0))
                t0 = time.perf_counter()
        if x0 is None:
            #if every single initialization had an infinite objective, just raise an error
            raise ValueError
        #return the initialized result
        return x0

    def _compute_weights(self):
        raise NotImplementedError
        
    def _objective(self, itr):
        raise NotImplementedError
        
    def _error(self):
        raise NotImplementedError
  
    def _get_mixture(self):
        raise NotImplementedError
