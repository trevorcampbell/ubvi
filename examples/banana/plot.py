import autograd.numpy as np
import pickle as pk
import bokeh.plotting as bkp
import bokeh.layouts as bkl

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from common import mixture_logpdf, preprocess_plot, postprocess_plot, pal, logFmtr
from ubvi.autograd import logsumexp

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

#load results
f = open('results/banana.pk', 'rb')
ubvis, bbvis, bbvi2s = pk.load(f)
f.close()

N_runs = len(ubvis)
N = ubvis[0][0].shape[0]

fkl_ubvi = np.zeros((N_runs, N))
rkl_ubvi = np.zeros((N_runs, N))
cput_ubvi = np.zeros((N_runs, N))
fkl_bbvi = np.zeros((N_runs, N))
rkl_bbvi = np.zeros((N_runs, N))
cput_bbvi = np.zeros((N_runs, N))
fkl_bbvi2 = np.zeros((N_runs, N))
rkl_bbvi2 = np.zeros((N_runs, N))
cput_bbvi2 = np.zeros((N_runs, N))
for i in range(len(ubvis)):
    mus = ubvis[i]['mus']
    Sigs = ubvis[i]['Sigs']
    wts = ubvis[i]['weights']
    cput_ubvi[i,:] = ubvis[i]['cputs']
    fkl_ubvi[i,:] = kldiv(mus, Sigs, wts, logp, p_sample, direction='forward')
    rkl_ubvi[i,:] = kldiv(mus, Sigs, wts, logp, p_sample, direction='reverse')
    cput_ubvi[i,:] = cput
    
    mus = bbvis[i]['mus']
    Sigs = bbvis[i]['Sigs']
    wts = bbvis[i]['weights']
    cput_bbvi[i,:] = bbvis[i]['cputs']
    fkl_bbvi[i,:] = kldiv(mus, Sigs, wts, logp, p_sample, direction='forward')
    rkl_bbvi[i,:] = kldiv(mus, Sigs, wts, logp, p_sample, direction='reverse')
    cput_bbvi[i,:] = cput

    mus = bbvi2s[i]['mus']
    Sigs = bbvi2s[i]['Sigs']
    wts = bbvi2s[i]['weights']
    cput_bbvi2[i,:] = bbvi2s[i]['cputs']
    fkl_bbvi2[i,:] = kldiv(mus, Sigs, wts, logp, p_sample, direction='forward')
    rkl_bbvi2[i,:] = kldiv(mus, Sigs, wts, logp, p_sample, direction='reverse')
    cput_bbvi2[i,:] = cput

print('CPU Time per component:')
print('UBVI: ' + str(cput_ubvi.mean()) + '+/-' + str(cput_ubvi.std()))
print('BVI: ' + str(cput_bbvi.mean()) + '+/-' + str(cput_bbvi.std()))
print('BVI2: ' + str(cput_bbvi2.mean()) + '+/-' + str(cput_bbvi2.std()))

plot_idx = 1
u_mu = ubvis[plot_idx]['mus']
u_Sig = ubvis[plot_idx]['Sigs']
u_wt = ubvis[plot_idx]['weights']

b_mu = bbvis[plot_idx]['mus']
b_Sig = bbvis[plot_idx]['Sigs']
b_wt = bbvis[plot_idx]['weights']

b2_mu = bbvi2s[plot_idx]['mus']
b2_Sig = bbvi2s[plot_idx]['Sigs']
b2_wt = bbvi2s[plot_idx]['weights']

#plot the contours
x = np.linspace(-30, 30, 600)
y = np.linspace(-50, 20, 700)
xx, yy = np.meshgrid(x, y)
x = xx.reshape(-1,1)
y = yy.reshape(-1,1)
X = np.hstack((x,y))
#plot the truth
Y = np.exp(banana(X)).reshape(700,600)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors='black', linewidths=2) #cmap="Blues_r")

#plot UBVI
lq_sqrt = 0.5*mvnlogpdf(X, u_mu, u_Sig, u_Siginv)
lq = 2*logsumexp(lq_sqrt + np.log(u_lmb), axis=1)
Y = np.exp(lq).reshape(700,600)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[0], linewidths=2) #cmap="Dark2")


#plot BVI
lq= mvnlogpdf(X, b_mu, b_Sig, b_Siginv)
lq = logsumexp(lq + np.log(b_w), axis=1)
Y = np.exp(lq).reshape(700,600)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[1], linewidths=2) #cmap="Dark2")

#plot BVI
lq= mvnlogpdf(X, b2_mu, b2_Sig, b2_Siginv)
lq = logsumexp(lq + np.log(b2_w), axis=1)
Y = np.exp(lq).reshape(700,600)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[2], linewidths=2) #cmap="Dark2")

plt.show()


#plot the KL vs iteration
fig2 = bkp.figure(width=1000,height=500,x_axis_label='# Components', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig2, '42pt', True)
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
preprocess_plot(fig3, '42pt', True)
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

#plot the KL vs iteration
fig4 = bkp.figure(width=1000,height=500,x_axis_label='# Components', y_axis_label='KL(q || p)', y_axis_type='log')
preprocess_plot(fig4, '42pt', True)
fig4.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 50, axis=0), line_width=6.5, color=pal[0])#, legend='UBVI')
fig4.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 25, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig4.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 75, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig4.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 50, axis=0), line_width=6.5, color=pal[1])#, legend='BVI(?)')
fig4.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 25, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig4.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 75, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig4.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BVI(??)')
fig4.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
fig4.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')

#plot the KL vs cput
fig5 = bkp.figure(width=1000,height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(q || p)', y_axis_type='log')
preprocess_plot(fig5, '42pt', True)
for cput, kl, nm, clrid in [(cput_ubvi, rkl_ubvi, 'UBVI', 0), (cput_bbvi, rkl_bbvi, 'BVI1', 1), (cput_bbvi2, rkl_bbvi2, 'BVI70', 2)]:
  cput_25 = np.percentile(np.cumsum(cput, axis=1), 25, axis=0)
  cput_50 = np.percentile(np.cumsum(cput, axis=1), 50, axis=0)
  cput_75 = np.percentile(np.cumsum(cput, axis=1), 75, axis=0)
  rkl_25 = np.percentile(kl, 25, axis=0)
  rkl_50 = np.percentile(kl, 50, axis=0)
  rkl_75 = np.percentile(kl, 75, axis=0)
  fig5.circle(cput_50, rkl_50, color=pal[clrid], size=10)#, legend=nm)
  fig5.segment(x0=cput_50, y0=rkl_25, x1=cput_50, y1=rkl_75, color=pal[clrid], line_width=4)#, legend=nm)
  fig5.segment(x0=cput_25, y0=rkl_50, x1=cput_75, y1=rkl_50, color=pal[clrid], line_width=4)#, legend=nm)


postprocess_plot(fig, '42pt')
postprocess_plot(fig2, '42pt')
postprocess_plot(fig3, '42pt')
postprocess_plot(fig4, '42pt')
postprocess_plot(fig5, '42pt')



bkp.show(bkl.gridplot([[fig2, fig3, fig4, fig5]]))

