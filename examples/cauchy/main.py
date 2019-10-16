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
n_samples = 500
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 1./np.sqrt(itr+1)
adam_num_iters = 10000
n_init = 10000
init_inflation = 16
lmb_good = lambda itr : 1./(1+itr)
lmb_bad = lambda itr : 70./(1+itr)

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

    #BBVI Good
    bbvi_good = BBVI(logp, gauss, adam, lmb = lmb_good, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    bbvi_good_results = []
    for n in range(1,N+1):
        bbvi_good_results.append(bbvi_good.build(n))

    #BBVI Bad
    bbvi_bad = BBVI(logp, gauss, adam, lmb = lmb_bad, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    bbvi_bad_results = []
    for n in range(1,N+1):
        bbvi_bad_results.append(bbvi_bad.build(n))

    if os.path.exists('results/cauchy.pk'):
        f = open('results/cauchy.pk', 'rb')
        res = pk.load(f)
        f.close()
        res[0].append(ubvi_results)
        res[1].append(bbvi_good_results)
        res[2].append(bbvi_bad_results)
        f = open('results/cauchy.pk', 'wb')
        pk.dump(res, f)
        f.close()
    else:
        f = open('results/cauchy.pk', 'wb')
        pk.dump(([ubvi_results], [bbvi_good_results], [bbvi_bad_results]), f)
        f.close()
