class Component(object):
    
    def __init__(self, d):
        self.d = d
        self.dim = None
    
    def reparam(self, Params):
        raise NotImplementedError
        
    def logpdf(self, Params, X):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
    
    def log_sqrt_pair_integral(self):
        raise NotImplementedError




