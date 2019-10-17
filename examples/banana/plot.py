import numpy as np
import pickle as pk
import bokeh.plotting as bkp
import bokeh.layouts as bkl
from scipy.stats import cauchy as sp_cauchy
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from common import mixture_logpdf, preprocess_plot, postprocess_plot, pal, logFmtr, kl_estimate, mixture_sample

def logp(X):
    X=np.atleast_2d(X)
    b = 0.1
    x = X[:,0]
    y = X[:,1]
    return -x**2/200 - (y+b*x**2-100*b)**2/2 - np.log(2*np.pi*10)

def p_sample(n_samples):
    b = 0.1
    X = np.random.multivariate_normal(np.zeros(2), np.array([[100, 0], [0, 1]]), n_samples)
    X[:, 1] = X[:, 1] - b*X[:, 0]**2 + 100*b
    return X

p_samps = p_sample(10000)

#load results
f = open('results/banana.pk', 'rb')
ubvis, bbvis, bbviepss = pk.load(f)
f.close()

N_runs = len(ubvis)
N = len(ubvis[0])

fkl_ubvi = np.zeros((N_runs, N))
cput_ubvi = np.zeros((N_runs, N))
fkl_bbvi = np.zeros((N_runs, N))
cput_bbvi = np.zeros((N_runs, N))
fkl_bbvieps = np.zeros((N_runs, N))
cput_bbvieps = np.zeros((N_runs, N))

for i in range(len(ubvis)):
    print('UBVI ' + str(i))
    cput_ubvi[i,:] = np.array([ ubvis[i][n]['cput'] for n in range(len(ubvis[i]))])
    fkl_ubvi[i,:] = np.array([kl_estimate(ubvis[i][n]['mus'], ubvis[i][n]['Sigs'], ubvis[i][n]['weights'], logp, p_samps, direction='forward') for n in range(len(ubvis[i]))])
    
    print('BVI ' + str(i))
    cput_bbvi[i,:] = np.array([ bbvis[i][n]['cput'] for n in range(len(bbvis[i]))])
    fkl_bbvi[i,:] = np.array([kl_estimate(bbvis[i][n]['mus'], bbvis[i][n]['Sigs'], bbvis[i][n]['weights'], logp, p_samps, direction='forward') for n in range(len(bbvis[i]))])

    print('BVI Eps ' + str(i))
    cput_bbvieps[i,:] = np.array([ bbviepss[i][n]['cput'] for n in range(len(bbviepss[i]))])
    fkl_bbvieps[i,:] = np.array([kl_estimate(bbviepss[i][n]['mus'], bbviepss[i][n]['Sigs'], bbviepss[i][n]['weights'], logp, p_samps, direction='forward') for n in range(len(bbviepss[i]))])

print('CPU Time per component:')
print('UBVI: ' + str(np.diff(cput_ubvi, axis=1).mean()) + '+/-' + str(np.diff(cput_ubvi, axis=1).std()))
print('BVI: ' + str(np.diff(cput_bbvi, axis=1).mean()) + '+/-' + str(np.diff(cput_bbvi, axis=1).std()))
print('BVI Eps: ' + str(np.diff(cput_bbvieps, axis=1).mean()) + '+/-' + str(np.diff(cput_bbvieps, axis=1).std()))

plot_idx = 0
plot_N = 29
u_mu = ubvis[plot_idx][plot_N]['mus']
u_Sig = ubvis[plot_idx][plot_N]['Sigs']
u_wt = ubvis[plot_idx][plot_N]['weights']

b_mu = bbvis[plot_idx][plot_N]['mus']
b_Sig = bbvis[plot_idx][plot_N]['Sigs']
b_wt = bbvis[plot_idx][plot_N]['weights']

beps_mu = bbviepss[plot_idx][plot_N]['mus']
beps_Sig = bbviepss[plot_idx][plot_N]['Sigs']
beps_wt = bbviepss[plot_idx][plot_N]['weights']

#plot the contours
wg = 120
hg = 140
x = np.linspace(-30, 30, wg)
y = np.linspace(-50, 20, hg)
xx, yy = np.meshgrid(x, y)
x = xx.reshape(-1,1)
y = yy.reshape(-1,1)
X = np.hstack((x,y))
#plot the truth
Y = np.exp(logp(X)).reshape(hg,wg)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors='black', linewidths=2) #cmap="Blues_r")

#plot UBVI
lq = mixture_logpdf(X, u_mu, u_Sig, u_wt)
Y = np.exp(lq).reshape(hg,wg)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[0], linewidths=2) #cmap="Dark2")


#plot BVI
lq = mixture_logpdf(X, b_mu, b_Sig, b_wt)
Y = np.exp(lq).reshape(hg,wg)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[1], linewidths=2) #cmap="Dark2")

#plot BVI Eps
lq = mixture_logpdf(X, beps_mu, beps_Sig, beps_wt)
Y = np.exp(lq).reshape(hg,wg)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[2], linewidths=2) #cmap="Dark2")

plt.show()


#plot the KL vs iteration
fig2 = bkp.figure(width=1000,height=500,x_axis_label='# Components', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig2, '42pt', log_scale_y = True)
fig2.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 50, axis=0), line_width=6.5, color=pal[0])#, legend='UBVI')
fig2.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 25, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig2.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 75, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig2.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 50, axis=0), line_width=6.5, color=pal[1])#, legend='BVI-1/(n+1)')
fig2.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 25, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig2.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 75, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig2.line(np.arange(fkl_bbvieps.shape[1])+1, np.percentile(fkl_bbvieps, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BVI-70/(n+1)')
fig2.line(np.arange(fkl_bbvieps.shape[1])+1, np.percentile(fkl_bbvieps, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
fig2.line(np.arange(fkl_bbvieps.shape[1])+1, np.percentile(fkl_bbvieps, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')

#plot the KL vs cput
fig3 = bkp.figure(width=1000,height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig3, '42pt', log_scale_x = True, log_scale_y = True)
#for cput, kl, nm, clrid in [(cput_ubvi, fkl_ubvi, 'UBVI', 0), (cput_bbvi, fkl_bbvi, 'BVI1', 1)]:
for cput, kl, nm, clrid in [(cput_ubvi, fkl_ubvi, 'UBVI', 0), (cput_bbvi, fkl_bbvi, 'BVI1', 1), (cput_bbvieps, fkl_bbvieps, 'BVIEps', 2)]:
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
#fig4.line(np.arange(rkl_bbvieps.shape[1])+1, np.percentile(rkl_bbvieps, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BVI(??)')
#fig4.line(np.arange(rkl_bbvieps.shape[1])+1, np.percentile(rkl_bbvieps, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
#fig4.line(np.arange(rkl_bbvieps.shape[1])+1, np.percentile(rkl_bbvieps, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
#
##plot the KL vs cput
#fig5 = bkp.figure(width=1000,height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(q || p)', y_axis_type='log')
#preprocess_plot(fig5, '42pt', True)
#for cput, kl, nm, clrid in [(cput_ubvi, rkl_ubvi, 'UBVI', 0), (cput_bbvi, rkl_bbvi, 'BVI1', 1), (cput_bbvieps, rkl_bbvieps, 'BVI70', 2)]:
#  cput_25 = np.percentile(np.cumsum(cput, axis=1), 25, axis=0)
#  cput_50 = np.percentile(np.cumsum(cput, axis=1), 50, axis=0)
#  cput_75 = np.percentile(np.cumsum(cput, axis=1), 75, axis=0)
#  rkl_25 = np.percentile(kl, 25, axis=0)
#  rkl_50 = np.percentile(kl, 50, axis=0)
#  rkl_75 = np.percentile(kl, 75, axis=0)
#  fig5.circle(cput_50, rkl_50, color=pal[clrid], size=10)#, legend=nm)
#  fig5.segment(x0=cput_50, y0=rkl_25, x1=cput_50, y1=rkl_75, color=pal[clrid], line_width=4)#, legend=nm)
#  fig5.segment(x0=cput_25, y0=rkl_50, x1=cput_75, y1=rkl_50, color=pal[clrid], line_width=4)#, legend=nm)


#postprocess_plot(fig, '42pt')
postprocess_plot(fig2, '42pt')
postprocess_plot(fig3, '42pt')
#postprocess_plot(fig4, '42pt')
#postprocess_plot(fig5, '42pt')



#bkp.show(bkl.gridplot([[fig, fig2, fig3, fig4, fig5]]))
bkp.show(bkl.gridplot([[fig2, fig3]]))







