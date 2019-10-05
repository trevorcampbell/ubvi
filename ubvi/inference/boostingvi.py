import autograd.numpy as np
from autograd import grad
import time
from ..optimization.adam import adam


class BoostingVI(object):
    
    def __init__(self, component_dist, opt_alg, n_init = 10, init_inflation = 100, estimate_error = True, verbose = True):
        self.N = 0 #current num of components
        self.component_dist = component_dist #component distribution object
        self.opt_alg = opt_alg #optimization algorithm function
        self.weights = [] #trace of weights
        self.params = np.empty((0, 0)) 
        self.cputs = [] #list of computation times for each build step
        self.errors = [] #list of error estimates after each build step
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
            new_param = self.opt_alg(x0, self._objective, grd)
            if self.verbose: print("Optimization of component " + str(i+1) + " complete")

            #add it to the matrix of flattened parameters
            self.params = np.vstack((self.params, new_param))

            #compute the new weights and add to the list
            if self.verbose: print('Updating weights...')
            self.weights.append(np.atleast_1d(self._compute_weights()))
            if self.verbose: print('Weight update complete...')

            #compute the time taken for this step
            self.cputs.append(time.perf_counter() - t0)

            #estimate current error if desired
            if self.estimate_error:
                err_name, err_val = self._error()
                self.errors.append(err_val)

            #print out the current error
            if self.verbose:
                print('Component ' + str(self.params.shape[0]) +':')
                print('CPU Time: ' + str(self.cputs[-1]))
                if self.estimate_error:
                    print(err_name +': ' + str(err_val))
                print('Params:' + str(self.component_dist.unflatten(self.params)))
                print('Weights: ' + str(self.weights[-1]))
            
        #update self.N to the new # comps
        self.N = N

        #generate the nicely-formatted output params
        output = self.component_dist.unflatten(self.params)
        #add weights, instrumentation (e.g. cput)
        output.update([('weights', self.weights), ('cputs', self.cputs)])
	
        return output
        
    def _initialize(self):
        x0 = None
        obj0 = np.inf
        #try initializing n_init times
        for n in range(self.n_init):
            xtmp = self.component_dist.params_init(self.params, self.weights, self.init_inflation)
            objtmp = self._objective(xtmp, -1)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                if self.verbose: print('Current best initialization -- x0: ' + str(x0) + ' obj0 = ' + str(obj0))
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
