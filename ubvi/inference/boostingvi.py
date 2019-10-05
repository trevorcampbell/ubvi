import autograd.numpy as np
from autograd import grad
import time


class BoostingVI(object):
    
    def __init__(self, target, component_dist, opt_alg, n_init = 10, init_inflation = 100, estimate_error = True, verbose = True):
        self.target = target #the target for the boosting algorithm (e.g. log density)
        self.N = 0 #current num of components
        self.component_dist = component_dist #component distribution object
        self.opt_alg = opt_alg #optimization algorithm function
        self.weights = [] #trace of weights
        self.params = np.empty((0, 0)) 
        self.cputs = [] #list of computation times for each build step
        self.errors = [] #list of error estimates after each build step
        self.n_init = n_init #number of times to initialize each component
        self.init_inflation = init_inflation #number of times to initialize each component
        
    def build(self, N):
	#build the approximation up to N components
        for i in range(self.N, N):
            t0 = time.process_time()

            #build the objective function and obtain its gradient via autodiff
            obj = lambda x, itr: self._objective(x, itr)
            grd = grad(obj)

            #initialize the next component
            if self.verbose: print("Initializing component " + str(n) +"... ")
            x0 = self._initialize(obj)
            if self.verbose: print("Initialization of component " + str(n)+ " complete, x0 = " + str(x0))
            
            #build the next component
            if self.verbose: print("Optimizing component " + str(n) +"... ")
            new_param = self.opt_alg(x0, obj, grd, callback=lambda prms, itr, grd : self.component_dist.print_perf(prms, itr, grd, self.print_every, obj))
            if self.verbose: print("Optimization of component " + str(n) + " complete")

            #add it to the matrix of flattened parameters
            self.params = np.vstack((self.params, new_param))

            #compute the new weights and add to the list
            if self.verbose: print('Updating weights...')
            self.weights.append(np.atleast_1d(self._compute_weights()))
            if self.verbose: print('Weight update complete...')

            #compute the time taken for this step
            self.cputs.append(time.process_time() - t0)

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
        output.update([('weights', self.weights), ('cput', self.cput)])
	
        return output
        
    def _initialize(self, obj):
        x0 = None
        obj0 = np.inf
        #try initializing n_init times
        for n in range(self.n_init):
            xtmp = self.component_dist.params_init(self.params, self.weights, self.init_inflation)
            objtmp = obj(xtmp, -1)
            if objtmp < obj0:
                x0 = xtmp
                obj0 = objtmp
                if self.verbose: print('Current best initialization -- x0: ' + str(x0) + ' obj0 = ' + str(obj0))
        if x0 is None:
            #if every single initialization had an infinite objective, just raise an error
            raise ValueError
        #return the initialized result
        return x0
        
    def _objective(self, itr):
        raise NotImplementedError
        
    def _compute_weights(self):
        raise NotImplementedError
    
    def _error(self):
        raise NotImplementedError
