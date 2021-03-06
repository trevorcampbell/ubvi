import autograd.numpy as np
import pickle as pk
import os

from ubvi.components import Gaussian
from ubvi.optimization import adam as ubvi_adam
from ubvi.inference import UBVI, BBVI
from ubvi.autograd import logsumexp

def logp(x):
    return (- np.log(1 + x**2) - np.log(np.pi)).flatten()

np.random.seed(1)

N_runs = 20
N = 30
d = 1
diag = True
n_samples = 2000
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 10./np.sqrt(itr+1)
adam_num_iters = 10000
n_init = 10000
init_inflation = 16
lmb = lambda itr : 1./np.sqrt(1+itr)

gauss = Gaussian(d, diag)
adam = lambda x0, obj, grd : ubvi_adam(x0, obj, grd, adam_learning_rate, adam_num_iters, callback = gauss.print_perf)


if not os.path.exists('results/'):
    os.mkdir('results')

for i in range(N_runs):
    print('Run ' + str(i+1)+'/'+str(N_runs))

    #UBVI
    ubvi = UBVI(logp, gauss, adam, n_init = n_init, n_samples = n_samples, n_logfg_samples = n_logfg_samples, init_inflation = init_inflation)
    ubvi_results = []
    for n in range(1,N+1):
        ubvi_results.append(ubvi.build(n))

    #BBVI 
    bbvi = BBVI(logp, gauss, adam, lmb = lmb, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    bbvi_results = []
    for n in range(1,N+1):
        bbvi_results.append(bbvi.build(n))

    ##BBVI eps
    bbvi_eps = BBVI(logp, gauss, adam, lmb = lmb, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation, eps=1e-3)
    bbvi_eps_results = []
    for n in range(1,N+1):
        bbvi_eps_results.append(bbvi_eps.build(n))

    if os.path.exists('results/cauchy.pk'):
        f = open('results/cauchy.pk', 'rb')
        res = pk.load(f)
        f.close()
        res[0].append(ubvi_results)
        res[1].append(bbvi_results)
        res[2].append(bbvi_eps_results)
        f = open('results/cauchy.pk', 'wb')
        pk.dump(res, f)
        f.close()
    else:
        f = open('results/cauchy.pk', 'wb')
        pk.dump(([ubvi_results], [bbvi_results], [bbvi_eps_results]), f)
        f.close()
