class Component(object):
    
    def unflatten(self, params):
        raise NotImplementedError

    def logpdf(self, params, X):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
    
    def log_sqrt_pair_integral(self):
        raise NotImplementedError




