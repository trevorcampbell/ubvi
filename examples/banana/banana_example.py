import autograd.numpy as np
import pickle as pk
import os

from distributions import Gaussian
from ubvi import UBVI
from bbvi import BBVI


def banana(X):
    b = 0.1
    x = X[:,0]
    y = X[:,1]
    return -x**2/200 - (y+b*x**2-100*b)**2/2 - np.log(2*np.pi*10)

def logf(x):
   return 0.5*banana(x)



N_runs = 1
N = 3
d = 2
diag = False
n_samples = 100
n_logfg_samples = 10000
adam_num_iters = 3000
adam_learning_rate = lambda itr : 0.1/np.sqrt(1.+itr)
n_init=1000
lmb_good = lambda itr : 70./(itr+1)
lmb_bad = lambda itr : 1./(itr+1)
print_every=100

if not os.path.exists('results/'):
  os.mkdir('results')

gauss = Gaussian(d, diag)

for i in range(N_runs):
  print('RUN ' + str(i+1)+'/'+str(N_runs))
  ubvi = UBVI(logf, N, gauss, n_samples, n_logfg_samples, n_init, adam_learning_rate, adam_num_iters, print_every)
  banana_ubvi = ubvi.build()

  bbvi = BBVI(banana, N, gauss, n_samples, n_init, lmb_good, adam_learning_rate, adam_num_iters, print_every)
  banana_bbvi = bbvi.build()

  bbvi2 = BBVI(banana, N, gauss, n_samples, n_init, lmb_bad, adam_learning_rate, adam_num_iters, print_every)
  banana_bbvi2 = bbvi2.build()

  if os.path.exists('results/banana.pk'):
      f = open('results/banana.pk', 'rb')
      res = pk.load(f)
      f.close()
      res[0].append(banana_ubvi)
      res[1].append(banana_bbvi)
      res[2].append(banana_bbvi2)
      f = open('results/banana.pk', 'wb')
      pk.dump(res, f)
      f.close()
  else:
      f = open('results/banana.pk', 'wb')
      pk.dump((banana_ubvi, banana_bbvi, banana_bbvi2), f)
      f.close()