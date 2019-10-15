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

N_runs = 2	
N = 30
d = 1
diag = True
n_samples = 500
n_logfg_samples = 10000
adam_learning_rate= lambda itr : 1./np.sqrt(itr+1)
adam_num_iters = 3000
n_init = 10000
init_inflation = 100
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
    cauchy_ubvi = ubvi.build(N)

    #BBVI Good
    bbvi_good = BBVI(logp, gauss, adam, lmb = lmb_good, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    cauchy_bbvi_good = bbvi_good.build(N)

    #BBVI Bad
    bbvi_bad = BBVI(logp, gauss, adam, lmb = lmb_bad, n_init = n_init, n_samples = n_samples, init_inflation = init_inflation)
    cauchy_bbvi_bad = bbvi_bad.build(N)

    if os.path.exists('results/cauchy.pk'):
        f = open('results/cauchy.pk', 'rb')
        res = pk.load(f)
        f.close()
        res[0].append(cauchy_ubvi)
        res[1].append(cauchy_bbvi_good)
        res[2].append(cauchy_bbvi_bad)
        f = open('results/cauchy.pk', 'wb')
        pk.dump(res, f)
        f.close()
    else:
        f = open('results/cauchy.pk', 'wb')
        pk.dump(([cauchy_ubvi], [cauchy_bbvi_good], [cauchy_bbvi_bad]), f)
        f.close()
