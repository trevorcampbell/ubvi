import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy import stats
import pickle as pk
import os

from UBVI import ubvi
from BBVI import bbvi




def logp(x):
    lw = np.log(np.array([0.5, 0.5]))
    lf = np.hstack(( stats.multivariate_normal.logpdf(x, 0, np.atleast_2d(0.5))[:,np.newaxis], stats.multivariate_normal.logpdf(x, 25, np.atleast_2d(5))[:,np.newaxis]))
    return logsumexp(lf + lw, axis=1)



def logf(x):
    return 0.5*logp(x)



N = 2
d = 1
Diag = False
n_samples = 500
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 0.1/np.sqrt(itr+1)
adam_num_iters = 3000
n_init=1000



ubvi_ = ubvi(logf, N, d, Diag, n_samples, n_logfg_samples, adam_learning_rate, adam_num_iters, n_init=n_init)


lmb = lambda itr : 10
bbvi_10 = bbvi(logp, N, d, Diag, n_samples, lmb, adam_learning_rate, adam_num_iters, n_init=n_init)


lmb = lambda itr : 1.
bbvi_1 = bbvi(logp, N, d, Diag, n_samples, lmb, adam_learning_rate, adam_num_iters, n_init=n_init)


lmb = lambda itr : 30.
bbvi_30 = bbvi(logp, N, d, Diag, n_samples, lmb, adam_learning_rate, adam_num_iters, n_init=n_init)


if not os.path.exists('results/'):
    os.mkdir('results')


f = open('results/mixture_results.pk', 'wb')
pk.dump(([bbvi_1, bbvi_10, bbvi_30], [1,10, 30], ubvi_), f)
f.close()

   
