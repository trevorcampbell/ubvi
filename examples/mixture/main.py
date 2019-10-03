import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats
import pickle as pk
import os

from ubvi.components import Gaussian
from ubvi.optimization import adam
from ubvi import UBVI
from bbvi import BBVI


def logp(x):
    lw = np.log(np.array([0.5, 0.5]))
    lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis]))
    return logsumexp(lf + lw, axis=1)

def logf(x):
    return 0.5*logp(x)


N = 3
d = 1
diag = False
n_samples = 500
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 0.1/np.sqrt(itr+1)
adam_num_iters = 3000
n_init = 1000

gauss = Gaussian(d, diag)
adam = lambda grd, x0, callback = None : (grd, x0, adam_learning_rate, adam_num_iters, callback)

ubvi = UBVI(logf, N, gauss, adam, n_samples, n_init, n_logfg_samples)
mixture_ubvi = ubvi.build()

lmb = lambda itr : 10
bbvi1 = BBVI(logp, N, gauss, adam, n_samples, n_init, lmb)
mixture_bbvi1 = bbvi1.build()

lmb = lambda itr : 1.
bbvi2 = BBVI(logp, N, gauss, adam, n_samples, n_init, lmb)
mixture_bbvi2 = bbvi2.build()

lmb = lambda itr : 30.
bbvi3 = BBVI(logp, N, gauss, adam, n_samples, n_init, lmb)
mixture_bbvi3 = bbvi3.build()


if not os.path.exists('results/'):
    os.mkdir('results')

f = open('results/mixture_results.pk', 'wb')
pk.dump(([mixture_bbvi1, mixture_bbvi2, mixture_bbvi3], [1,10, 30], mixture_ubvi), f)
f.close()
