import numpy as np
import bokeh.plotting as bkp
import pickle as pk
import autograd.scipy.stats as stats

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from common import mixture_logpdf, preprocess_plot, postprocess_plot, pal
from ubvi.autograd import logsumexp


#load the results
f = open('results/mixture_results.pk', 'rb')
res = pk.load(f)
f.close()

bbvis = res[0]
bbvi_lmbs = res[1]
ubvi = res[2]

X = np.linspace(-10,40,5000)

fig = bkp.figure()
preprocess_plot(fig, '24pt', False)
for i in range(len(bbvis)):
  #extract results
  mus = bbvis[i]['mus']
  Sigs = bbvis[i]['Sigs']
  weights = bbvis[i]['weights']
  lmb = bbvi_lmbs[i]
  #compute log density for KL boosting VI
  lg = mixture_logpdf(X[:, np.newaxis], mus, Sigs, weights)
  #plot
  fig.line(X, np.exp(0.5*lg), line_width=6.5, color=pal[i+1], legend='BVI'+str(lmb))

#plot true density
lg = mixture_logpdf(X[:, np.newaxis], np.array([[0.], [25.]]), np.array([ [[0.5]], [[5.]] ]), np.array([0.5, 0.5]))
#lg = mixture_logpdf(X[:, np.newaxis], np.array([[0.]]), np.array([ [[1.0]]]), np.array([1.0]))
fig.line(X, np.exp(0.5*lg), line_width=6.5, line_color='black', legend='p(x)') 

#compute log density for UBVI
mus = ubvi['mus']
Sigs = ubvi['Sigs']
weights = ubvi['weights']
lg = mixture_logpdf(X[:, np.newaxis], mus, Sigs, weights)
fig.line(X, np.exp(0.5*lg), line_width=6.5, color=pal[0], legend='UBVI', line_dash = [20, 20])

postprocess_plot(fig, '24pt')
bkp.show(fig)


