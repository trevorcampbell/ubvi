import numpy as np
import pickle as pk
import bokeh.plotting as bkp
import bokeh.layouts as bkl
from scipy.stats import cauchy as sp_cauchy

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from common import mixture_logpdf, preprocess_plot, postprocess_plot, pal, logFmtr, kl_estimate, mixture_sample

def logp(x):
    return (- np.log(1 + x**2) - np.log(np.pi)).flatten()

p_samps = np.random.standard_cauchy(10000)[:,np.newaxis]

#load results
f = open('results/cauchy.pk', 'rb')
ubvis, bbvis, bbvi2s = pk.load(f)
f.close()

N_runs = len(ubvis)
N = len(ubvis[0])

fkl_ubvi = np.zeros((N_runs, N))
cput_ubvi = np.zeros((N_runs, N))
fkl_bbvi = np.zeros((N_runs, N))
cput_bbvi = np.zeros((N_runs, N))
fkl_bbvi2 = np.zeros((N_runs, N))
cput_bbvi2 = np.zeros((N_runs, N))

for i in range(len(ubvis)):
    print('UBVI ' + str(i))
    cput_ubvi[i,:] = np.array([ ubvis[i][n]['cput'] for n in range(len(ubvis[i]))])
    fkl_ubvi[i,:] = np.array([kl_estimate(ubvis[i][n]['mus'], ubvis[i][n]['Sigs'], ubvis[i][n]['weights'], logp, p_samps, direction='forward') for n in range(len(ubvis[i]))])
    
    print('BBVI Good ' + str(i))
    cput_bbvi[i,:] = np.array([ bbvis[i][n]['cput'] for n in range(len(bbvis[i]))])
    fkl_bbvi[i,:] = np.array([kl_estimate(bbvis[i][n]['mus'], bbvis[i][n]['Sigs'], bbvis[i][n]['weights'], logp, p_samps, direction='forward') for n in range(len(bbvis[i]))])

    print('BBVI Bad ' + str(i))
    cput_bbvi2[i,:] = np.array([ bbvi2s[i][n]['cput'] for n in range(len(bbvi2s[i]))])
    fkl_bbvi2[i,:] = np.array([kl_estimate(bbvi2s[i][n]['mus'], bbvi2s[i][n]['Sigs'], bbvi2s[i][n]['weights'], logp, p_samps, direction='forward') for n in range(len(bbvi2s[i]))])

print('CPU Time per component:')
print('UBVI: ' + str(np.diff(cput_ubvi, axis=1).mean()) + '+/-' + str(np.diff(cput_ubvi, axis=1).std()))
print('BVI-Good: ' + str(np.diff(cput_bbvi, axis=1).mean()) + '+/-' + str(np.diff(cput_bbvi, axis=1).std()))
print('BVI-Bad: ' + str(np.diff(cput_bbvi2, axis=1).mean()) + '+/-' + str(np.diff(cput_bbvi2, axis=1).std()))

plot_idx = 0
plot_N = 29
u_mu = ubvis[plot_idx][plot_N]['mus']
u_Sig = ubvis[plot_idx][plot_N]['Sigs']
u_wt = ubvis[plot_idx][plot_N]['weights']

b_mu = bbvis[plot_idx][plot_N]['mus']
b_Sig = bbvis[plot_idx][plot_N]['Sigs']
b_wt = bbvis[plot_idx][plot_N]['weights']

b2_mu = bbvi2s[plot_idx][plot_N]['mus']
b2_Sig = bbvi2s[plot_idx][plot_N]['Sigs']
b2_wt = bbvi2s[plot_idx][plot_N]['weights']

#plot the fit
X = np.linspace(-20,20,4000)
fig = bkp.figure(width=1000, height=1000, x_range=(X.min(), X.max()))
preprocess_plot(fig, '42pt')
#plot the truth
fig.line(X, np.exp(logp(X)), line_width=6.5, color='black', legend='p(x)')
#plot BVIbad
lq = mixture_logpdf(X[:, np.newaxis], b2_mu, b2_Sig, b2_wt)
fig.line(X, np.exp(lq), line_width=6.5, color=pal[2], legend='BVI70')
#plot BVIgood
lq = mixture_logpdf(X[:, np.newaxis], b_mu, b_Sig, b_wt)
fig.line(X, np.exp(lq), line_width=6.5, color=pal[1], legend='BVI1')
#plot UBVI
lq = mixture_logpdf(X[:, np.newaxis], u_mu, u_Sig, u_wt)
fig.line(X, np.exp(lq), line_width=6.5, color=pal[0], legend='UBVI')

#plot UBVI samples
#samps = mixture_sample(u_mu, u_Sig, u_wt, 50000)
#hist, edges = np.histogram(samps, density=True, bins=100, range=(X.min(), X.max()))
#fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=pal[0], alpha=0.3, line_color='white')

##plot Cauchy samples
#X = np.random.standard_cauchy(50000)
#hist, edges = np.histogram(X, density=True, bins=1000, range = (-50, 50))
#fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='black', alpha=0.3, line_color='white')

#plot the KL vs iteration
fig2 = bkp.figure(width=1000,height=500,x_axis_label='# Components', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig2, '42pt', log_scale_y = True)
fig2.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 50, axis=0), line_width=6.5, color=pal[0])#, legend='UBVI')
fig2.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 25, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig2.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 75, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig2.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 50, axis=0), line_width=6.5, color=pal[1])#, legend='BVI-1/(n+1)')
fig2.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 25, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig2.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 75, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig2.line(np.arange(fkl_bbvi2.shape[1])+1, np.percentile(fkl_bbvi2, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BVI-70/(n+1)')
fig2.line(np.arange(fkl_bbvi2.shape[1])+1, np.percentile(fkl_bbvi2, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
fig2.line(np.arange(fkl_bbvi2.shape[1])+1, np.percentile(fkl_bbvi2, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')

#plot the KL vs cput
fig3 = bkp.figure(width=1000,height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig3, '42pt', log_scale_x = True, log_scale_y = True)
for cput, kl, nm, clrid in [(cput_ubvi, fkl_ubvi, 'UBVI', 0), (cput_bbvi, fkl_bbvi, 'BVI1', 1), (cput_bbvi2, fkl_bbvi2, 'BVI70', 2)]:
  cput_25 = np.percentile(np.cumsum(cput, axis=1), 25, axis=0)
  cput_50 = np.percentile(np.cumsum(cput, axis=1), 50, axis=0)
  cput_75 = np.percentile(np.cumsum(cput, axis=1), 75, axis=0)
  fkl_25 = np.percentile(kl, 25, axis=0)
  fkl_50 = np.percentile(kl, 50, axis=0)
  fkl_75 = np.percentile(kl, 75, axis=0)
  fig3.circle(cput_50, fkl_50, color=pal[clrid], size=10)#, legend=nm)
  fig3.segment(x0=cput_50, y0=fkl_25, x1=cput_50, y1=fkl_75, color=pal[clrid], line_width=4)#, legend=nm)
  fig3.segment(x0=cput_25, y0=fkl_50, x1=cput_75, y1=fkl_50, color=pal[clrid], line_width=4)#, legend=nm)

##plot the KL vs iteration
#fig4 = bkp.figure(width=1000,height=500,x_axis_label='# Components', y_axis_label='KL(q || p)', y_axis_type='log')
#preprocess_plot(fig4, '42pt', True)
#fig4.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 50, axis=0), line_width=6.5, color=pal[0])#, legend='UBVI')
#fig4.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 25, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
#fig4.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 75, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
#fig4.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 50, axis=0), line_width=6.5, color=pal[1])#, legend='BVI(?)')
#fig4.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 25, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
#fig4.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 75, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
#fig4.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BVI(??)')
#fig4.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
#fig4.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
#
##plot the KL vs cput
#fig5 = bkp.figure(width=1000,height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(q || p)', y_axis_type='log')
#preprocess_plot(fig5, '42pt', True)
#for cput, kl, nm, clrid in [(cput_ubvi, rkl_ubvi, 'UBVI', 0), (cput_bbvi, rkl_bbvi, 'BVI1', 1), (cput_bbvi2, rkl_bbvi2, 'BVI70', 2)]:
#  cput_25 = np.percentile(np.cumsum(cput, axis=1), 25, axis=0)
#  cput_50 = np.percentile(np.cumsum(cput, axis=1), 50, axis=0)
#  cput_75 = np.percentile(np.cumsum(cput, axis=1), 75, axis=0)
#  rkl_25 = np.percentile(kl, 25, axis=0)
#  rkl_50 = np.percentile(kl, 50, axis=0)
#  rkl_75 = np.percentile(kl, 75, axis=0)
#  fig5.circle(cput_50, rkl_50, color=pal[clrid], size=10)#, legend=nm)
#  fig5.segment(x0=cput_50, y0=rkl_25, x1=cput_50, y1=rkl_75, color=pal[clrid], line_width=4)#, legend=nm)
#  fig5.segment(x0=cput_25, y0=rkl_50, x1=cput_75, y1=rkl_50, color=pal[clrid], line_width=4)#, legend=nm)


postprocess_plot(fig, '42pt')
postprocess_plot(fig2, '42pt')
postprocess_plot(fig3, '42pt')
#postprocess_plot(fig4, '42pt')
#postprocess_plot(fig5, '42pt')



#bkp.show(bkl.gridplot([[fig, fig2, fig3, fig4, fig5]]))
bkp.show(bkl.gridplot([[fig, fig2, fig3]]))



