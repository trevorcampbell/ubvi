#import seaborn as sns
#import matplotlib.pyplot as plt
#import pandas as pd
from utils import *
import numpy as np
from bokeh.models import Span
import bokeh.plotting as bkp
import bokeh.layouts as bkl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def plot_meanstd(plot, x, means, sigs, color, linewidth, alpha, line_dash, nm):
  plot.line(x, means, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.patch(np.hstack((x, x[::-1])), np.hstack(( means-sigs/2, (means+sigs/2)[::-1] )), color=color, line_width=linewidth/2, line_dash=line_dash, alpha=alpha, legend=nm)


res = np.load('results.npz')

sigsqs = res['sigsqs']
logkls = res['logkls']
logconds = res['logconds']
ds = res['ds']
n_trials = res['n_trials']
true2dsigs = res['true2dsigs']
logimptcerrs = res['logimptcerrs']
nms = res['nms']

f_kl = bkp.figure(width=1000, height=1000, x_axis_label='Condition Number', y_axis_label='KL(p || q)', x_axis_type='log', y_axis_type='log')
f_err = bkp.figure(width=1000, height=1000, x_axis_label='Condition Number', y_axis_label='Relative Covariance Estimation Error', x_axis_type='log', y_axis_type='log')
f_ell = bkp.figure(width=1000, height=1000)
for f in [f_kl, f_err]:
  preprocess_plot(f, '32pt', True)
preprocess_plot(f_ell, '32pt', False)

##GP regression  to plot
#kernel = RBF(3., (2., 4.))
#gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1.)
#x = np.linspace(0., 11., 100)
#
## Fit to data using Maximum Likelihood Estimation of the parameters
#gp.fit(logconds[:, np.newaxis], logkls[0, :])
#means, sigs = gp.predict(x[:, np.newaxis], return_std=True)
#plot_meanstd(f, x, means, sigs+np.sqrt(np.fabs(gp.alpha)), pal[0], 5, 0.3, 'solid', 'Hellinger(q, p)')
#
#gp.fit(logconds[:, np.newaxis], logkls[1, :])
#means, sigs = gp.predict(x[:, np.newaxis], return_std=True)
#plot_meanstd(f, x, means, sigs+np.sqrt(np.fabs(gp.alpha)), pal[1], 5, 0.3, 'solid', 'ChiSq(q || p)')
#
#gp.fit(logconds[:, np.newaxis], logkls[2, :])
#means, sigs = gp.predict(x[:, np.newaxis], return_std=True)
#plot_meanstd(f, x, means, sigs+np.sqrt(np.fabs(gp.alpha)), pal[2], 5, 0.3, 'solid', 'KL(q || p)')
#
#gp.fit(logconds[:, np.newaxis], logkls[3, :])
#means, sigs = gp.predict(x[:, np.newaxis], return_std=True)
#plot_meanstd(f, x, means, sigs+np.sqrt(np.fabs(gp.alpha)), pal[3], 5, 0.3, 'solid', 'KL(p || q)')


#
#
#
## Make the prediction on the meshed x-axis (ask for MSE as well)
#
#
#
#plot 1/2 hellinger and chisq at a time since they overlap a lot
idcs = np.arange(logkls.shape[1])
np.random.shuffle(idcs)
half = int(idcs.shape[0]/2)
third = int(idcs.shape[0]/1.5)
f_kl.scatter(10**logconds, 10**logkls[2, :], legend='KL(q || p)', color=pal[2], size=7) 
f_kl.scatter(10**logconds[idcs[:half]], 10**logkls[0, idcs[:half] ], legend='Hellinger(q, p)', color=pal[1], size=7) 
f_kl.scatter(10**logconds, 10**logkls[3, :], legend='KL(p || q)', color=pal[0], size=7) 
f_kl.scatter(10**logconds[idcs[half:]], 10**logkls[0, idcs[half:] ], legend='Hellinger(q, p)', color=pal[1], size=7) 
#f_kl.scatter(10**logconds, 10**logkls[1, :], legend='ChiSq(q || p)', color=pal[1], size=7) 
#f_kl.scatter(10**logconds[idcs[third:]], 10**logkls[3, idcs[third:]], legend='KL(p || q)', color=pal[3], size=7) 


#print(list(zip(10**logconds[:100], np.arange(100))))
plot_idx = 4
#print(10**logconds[plot_idx])
#print(true2dsigs[plot_idx, :, :])

plot_gaussian(f_ell, np.zeros(2), true2dsigs[plot_idx,:,:], 'black', 2, 1., 'solid', 'True')
#plot_gaussian(f_ell, np.zeros(2), 4*true2dsigs[plot_idx,:,:], 'black', 2, 1., 'solid', 'True')
#plot_gaussian(f_ell, np.zeros(2), 9*true2dsigs[plot_idx,:,:], 'black', 2, 1., 'solid', 'True')

for k in [2, 3, 0]:
  plot_gaussian(f_ell, np.zeros(2), sigsqs[k, plot_idx]*np.eye(2), pal[k], 5, 1., 'solid', nms[k])
##plot_gaussian(f_ell, np.zeros(2), 4*sigsqs[2, plot_idx]*np.eye(2), pal[2], 5, 1., 'solid', 'KL(q || p)')
##plot_gaussian(f_ell, np.zeros(2), 9*sigsqs[2, plot_idx]*np.eye(2), pal[2], 5, 1., 'solid', 'KL(q || p)')
#
#plot_gaussian(f_ell, np.zeros(2), sigsqs[3, plot_idx]*np.eye(2), pal[3], 5, 1., 'solid', 'KL(p || q)')
##plot_gaussian(f_ell, np.zeros(2), 4*sigsqs[3, plot_idx]*np.eye(2), pal[3], 5, 1., 'solid', 'KL(p || q)')
##plot_gaussian(f_ell, np.zeros(2), 9*sigsqs[3, plot_idx]*np.eye(2), pal[3], 5, 1., 'solid', 'KL(p || q)')
#
#plot_gaussian(f_ell, np.zeros(2), sigsqs[0, plot_idx]*np.eye(2), pal[0], 5, 1., 'solid', 'Hellinger(q, p)')
##plot_gaussian(f_ell, np.zeros(2), 4*sigsqs[0, plot_idx]*np.eye(2), pal[0], 5, 1., 'solid', 'Hellinger(q, p)')
##plot_gaussian(f_ell, np.zeros(2), 9*sigsqs[0, plot_idx]*np.eye(2), pal[0], 5, 1., 'solid', 'Hellinger(q, p)')
#
#plot_gaussian(f_ell, np.zeros(2), sigsqs[1, plot_idx]*np.eye(2), pal[1], 5, 1., 'solid', 'ChiSq(q || p)')
##plot_gaussian(f_ell, np.zeros(2), 4*sigsqs[1, plot_idx]*np.eye(2), pal[1], 5, 1., 'solid', 'ChiSq(q || p)')
##plot_gaussian(f_ell, np.zeros(2), 9*sigsqs[1, plot_idx]*np.eye(2), pal[1], 5, 1., 'solid', 'ChiSq(q || p)')

f_err.line([0, 0], [0, 0], line_color=pal[2], line_width=10, legend=nms[2])
f_err.scatter(10**logconds[idcs[:half]], 10**(logimptcerrs[0, idcs[:half]] - logimptcerrs[2, idcs[:half]]), color=pal[0], size=7, legend=nms[0])
f_err.scatter(10**logconds, 10**(logimptcerrs[3,:] - logimptcerrs[2, :]), color=pal[1], size=7, legend=nms[3])
f_err.scatter(10**logconds[idcs[half:]], 10**(logimptcerrs[0, idcs[half:]] - logimptcerrs[2, idcs[half:]]), color=pal[0], size=7, legend=nms[0])
hline = Span(location=1, dimension='width', line_color=pal[2], line_width=10)
f_err.renderers.extend([hline])



for f in [f_kl, f_err]:
  postprocess_plot(f, '32pt', orientation='vertical', location='top_left', glyph_width=60, glyph_height=60 )
postprocess_plot(f_ell, '32pt', orientation='vertical', location='top_right', glyph_width=60, glyph_height=60 )



bkp.show(bkl.gridplot([[f_kl, f_ell, f_err]]))


#sns.set_style('whitegrid')
#data= pd.read_pickle('results.pkl')
##data['Log KL Divergence'] -= np.log10( data['Dimension'])
#
#data.loc[data['Divergence'] == 'KL', 'Divergence'] = 'KL(q || p)'
#f = plt.plot()

#ax = sns.violinplot(data=data, x='Dimension', y='Log KL Divergence', hue='Divergence', palette='Set2', split=True, scale='count', inner=None) #None)

#split the plots in two to help visualize overlap
#ax = sns.scatterplot(data=data, x='Log Condition Number', y='Log KL Divergence', hue='Divergence', palette='colorblind', edgecolor=None, s=1)
#ax.set(xlabel='Log10 Condition Number', ylabel = 'Log10 KL(p || q)')

#plt.show()
