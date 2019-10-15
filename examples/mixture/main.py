import autograd.numpy as np
from autograd.scipy import stats
import pickle as pk
import os

from ubvi.components import Gaussian
from ubvi.optimization import adam as ubvi_adam
from ubvi.inference import UBVI, BBVI
from ubvi.autograd import logsumexp


def logp(x):
    lw = np.log(np.array([0.5, 0.5]))
    lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis]))
    return logsumexp(lf + lw, axis=1)

#def logp(x):
#    return stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(1.0))

np.random.seed(1)

N = 3
d = 1
diag = True
n_samples = 500
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 1./np.sqrt(itr+1)
adam_num_iters = 3000
n_init = 10000
init_inflation = 100

gauss = Gaussian(d, diag)
adam = lambda x0, obj, grd : ubvi_adam(x0, obj, grd, adam_learning_rate, adam_num_iters, callback = gauss.print_perf)

#UBVI
ubvi = UBVI(logp, gauss, adam, n_init = n_init, n_samples = n_samples, n_logfg_samples = n_logfg_samples, init_inflation = init_inflation)
mixture_ubvi = ubvi.build(N)

#BBVI 1
lmb = lambda itr : 1.
bbvi1 = BBVI(logp, gauss, adam, lmb = lmb, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
mixture_bbvi1 = bbvi1.build(N)

#BBVI 10
lmb = lambda itr : 10.
bbvi2 = BBVI(logp, gauss, adam, lmb = lmb, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
mixture_bbvi2 = bbvi2.build(N)


#BBVI 30
lmb = lambda itr : 30.
bbvi3 = BBVI(logp, gauss, adam, lmb = lmb, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
mixture_bbvi3 = bbvi3.build(N)

if not os.path.exists('results/'):
    os.mkdir('results')

f = open('results/mixture_results.pk', 'wb')
pk.dump(([mixture_bbvi1, mixture_bbvi2, mixture_bbvi3], [1,10, 30], mixture_ubvi), f)
f.close()
