from bbvi import mvnlogpdf
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd import grad
import pickle as pk
import matplotlib.pyplot as plt
import bokeh.palettes as bkpl
from bokeh.models import Span, FuncTickFormatter
import bokeh.plotting as bkp
import bokeh.layouts as bkl

plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)

pal = bkpl.colorblind['Colorblind'][8]
pl = [pal[1], pal[0], pal[3]]
pl.extend(pal[4:8])
pal = pl
logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079']
if (Math.log10(tick) < 0){
  return '10\u207B'+trns[Math.round(Math.abs(Math.log10(tick)))];
} else {
  return '10'+trns[Math.round(Math.abs(Math.log10(tick)))];
}
""")



def preprocess_plot(fig, axis_font_size, log_scale):
  fig.xaxis.axis_label_text_font_size= axis_font_size
  fig.xaxis.major_label_text_font_size= axis_font_size
  fig.yaxis.axis_label_text_font_size= axis_font_size
  fig.yaxis.major_label_text_font_size= axis_font_size
  if log_scale:
    fig.yaxis.formatter = logFmtr
  #fig.toolbar.logo = None
  #fig.toolbar_location = None

def postprocess_plot(fig, legend_font_size, orientation='vertical', location='top_right', glyph_width=80, glyph_height=40):
  fig.legend.label_text_font_size= legend_font_size
  fig.legend.orientation=orientation
  fig.legend.location=location
  fig.legend.glyph_width=glyph_width
  fig.legend.glyph_height=glyph_height
  fig.legend.spacing=5
  fig.xgrid.grid_line_color=None
  fig.ygrid.grid_line_color=None


def banana(X):
    X=np.atleast_2d(X)
    b = 0.1
    x = X[:,0]
    y = X[:,1]
    return -x**2/200 - (y+b*x**2-100*b)**2/2 - np.log(2*np.pi*10)

def logf(x):
   return 0.5*banana(x)

def kldiv(Mu, Sigma, W, n_samples=10000, method="ubvi", direction="forward"):

    b = 0.1
    X = np.random.multivariate_normal(np.zeros(2), np.array([[100, 0], [0, 1]]), n_samples)
    X[:, 1] = X[:, 1] - b*X[:, 0]**2 + 100*b

    ###plot X to check
    ##x = np.linspace(-30, 30, 600)
    ##y = np.linspace(-50, 20, 700)
    ##xx, yy = np.meshgrid(x, y)
    ##x = xx.reshape(-1,1)
    ##y = yy.reshape(-1,1)
    ##X_contour = np.hstack((x,y))
    ###plot the truth
    ##Y_contour = np.exp(banana(X_contour)).reshape(700,600)
    ##Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
    ##plt.contour(xx, yy, Y_contour, levels=Levels, colors='black') #cmap="Blues_r")
    ##plt.scatter(X[:, 0], X[:, 1], color='blue')
    ##plt.scatter(forward_kl.X[:, 0], forward_kl.X[:, 1], color='red')
    ##plt.show()

    lp = banana(X)
    Siginv = np.linalg.inv(Sigma)
    K = Mu.shape[0]
    kl = np.zeros(K)
    for i in range(K):
        if method=="ubvi":
            lq_sqrt = 0.5*mvnlogpdf(X, Mu[:i+1], Sigma[:i+1], Siginv[:i+1])
            lq = 2*logsumexp(lq_sqrt + np.log(W[i, :i+1]), axis=1)
        if method=="bbvi":
            lq= mvnlogpdf(X, Mu[:i+1], Sigma[:i+1], Siginv[:i+1])
            lq = logsumexp(lq + np.log(W[i, :i+1]), axis=1)
        if direction == "reverse":
          wts = (lq - lp)
          wts -= wts.max()
          wts = np.exp(wts)/(np.exp(wts).sum())
          kl[i] = (wts*(lq-lp)).sum()
        else:
          kl[i] = (lp - lq).mean()
    return kl

    
###############################################################################

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
  u_mu, u_Sig, u_lmb, U_LMB, Z, cput = ubvis[i]
  #code for when Sig is diagonal
  #u_Sig_diag = np.zeros((u_Sig.shape[0], u_Sig.shape[1], u_Sig.shape[1]))
  #for j in range(u_Sig.shape[0]):
  #  for k in range(u_Sig.shape[1]):
  #    u_Sig_diag[j,k,k] = u_Sig[j,k]
  #u_Sig = u_Sig_diag
  fkl_ubvi[i,:] = kldiv(u_mu, u_Sig, U_LMB, method='ubvi', direction='forward')
  rkl_ubvi[i,:] = kldiv(u_mu, u_Sig, U_LMB, method='ubvi', direction='reverse')
  cput_ubvi[i,:] = cput
  b_mu, b_Sig, b_w, B_W, cput = bbvis[i]
  fkl_bbvi[i,:] = kldiv(b_mu, b_Sig, B_W, method='bbvi', direction='forward')
  rkl_bbvi[i,:] = kldiv(b_mu, b_Sig, B_W, method='bbvi', direction='reverse')
  cput_bbvi[i,:] = cput
  b2_mu, b2_Sig, b2_w, B2_W, cput = bbvi2s[i]
  fkl_bbvi2[i,:] = kldiv(b2_mu, b2_Sig, B2_W, method='bbvi', direction='forward')
  rkl_bbvi2[i,:] = kldiv(b2_mu, b2_Sig, B2_W, method='bbvi', direction='reverse')
  cput_bbvi2[i,:] = cput


print('CPU Time per component:')
print('UBVI: ' + str(cput_ubvi.mean()) + '+/-' + str(cput_ubvi.std()))
print('BBVI: ' + str(cput_bbvi.mean()) + '+/-' + str(cput_bbvi.std()))
print('BBVI2: ' + str(cput_bbvi2.mean()) + '+/-' + str(cput_bbvi2.std()))



plot_idx = 4
#plot_idx = 3 #severe failure of bbvi
u_mu, u_Sig, u_lmb, U_LMB, Z, cput = ubvis[plot_idx]
b_mu, b_Sig, b_w, B_W, cput = bbvis[plot_idx]
b2_mu, b2_Sig, b2_w, B2_W, cput = bbvi2s[plot_idx]
#code for when Sig is diagonal
#u_Sig_diag = np.zeros((u_Sig.shape[0], u_Sig.shape[1], u_Sig.shape[1]))
#for j in range(u_Sig.shape[0]):
#  for k in range(u_Sig.shape[1]):
#    u_Sig_diag[j,k,k] = u_Sig[j,k]
#u_Sig = u_Sig_diag
u_Siginv = np.linalg.inv(u_Sig)
b_Siginv = np.linalg.inv(b_Sig)
b2_Siginv = np.linalg.inv(b2_Sig)

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


#plot BBVI
lq= mvnlogpdf(X, b_mu, b_Sig, b_Siginv)
lq = logsumexp(lq + np.log(b_w), axis=1)
Y = np.exp(lq).reshape(700,600)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[1], linewidths=2) #cmap="Dark2")

#plot BBVI
lq= mvnlogpdf(X, b2_mu, b2_Sig, b2_Siginv)
lq = logsumexp(lq + np.log(b2_w), axis=1)
Y = np.exp(lq).reshape(700,600)
Levels = np.array([0.001, 0.0025, 0.005, 0.01, 0.015, 0.025])
Levels = np.array([0.001, .005, 0.015, 0.025])
plt.contour(xx, yy, Y, levels=Levels, colors=pal[2], linewidths=2) #cmap="Dark2")



plt.show()

#plot the KL
fig = bkp.figure(width=1000, height=500, x_axis_label='# Components', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig, '42pt', True)

fig.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 50, axis=0), line_width=6.5, color=pal[0])#, legend='UBVI')
fig.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 25, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig.line(np.arange(fkl_ubvi.shape[1])+1, np.percentile(fkl_ubvi, 75, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 50, axis=0), line_width=6.5, color=pal[1])#, legend='BBVI-70/(n+1)')
fig.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 25, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig.line(np.arange(fkl_bbvi.shape[1])+1, np.percentile(fkl_bbvi, 75, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig.line(np.arange(fkl_bbvi2.shape[1])+1, np.percentile(fkl_bbvi2, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BBVI-1/(n+1)')
fig.line(np.arange(fkl_bbvi2.shape[1])+1, np.percentile(fkl_bbvi2, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
fig.line(np.arange(fkl_bbvi2.shape[1])+1, np.percentile(fkl_bbvi2, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')


#plot the KL vs cput
fig2 = bkp.figure(width=1000, height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(p || q)', y_axis_type='log')
preprocess_plot(fig2, '42pt', True)
for cput, kl, nm, clrid in [(cput_ubvi, fkl_ubvi, 'UBVI', 0), (cput_bbvi, fkl_bbvi, 'BBVI-70/(n+1)', 1), (cput_bbvi2, fkl_bbvi2, 'BBVI-1/(n+1)', 2)]:
  cput_25 = np.percentile(np.cumsum(cput, axis=1), 25, axis=0)
  cput_50 = np.percentile(np.cumsum(cput, axis=1), 50, axis=0)
  cput_75 = np.percentile(np.cumsum(cput, axis=1), 75, axis=0)
  fkl_25 = np.percentile(kl, 25, axis=0)
  fkl_50 = np.percentile(kl, 50, axis=0)
  fkl_75 = np.percentile(kl, 75, axis=0)
  fig2.circle(cput_50, fkl_50, color=pal[clrid], size=10)#, legend=nm)
  fig2.segment(x0=cput_50, y0=fkl_25, x1=cput_50, y1=fkl_75, color=pal[clrid], line_width=4)#, legend=nm)
  fig2.segment(x0=cput_25, y0=fkl_50, x1=cput_75, y1=fkl_50, color=pal[clrid], line_width=4)#, legend=nm)


#plot the KL
fig3 = bkp.figure(width=1000, height=500,x_axis_label='# Components', y_axis_label='KL(q || p)', y_axis_type='log')
preprocess_plot(fig3, '42pt', True)
fig3.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 50, axis=0), line_width=6.5, color=pal[0])#, legend='UBVI')
fig3.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 25, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig3.line(np.arange(rkl_ubvi.shape[1])+1, np.percentile(rkl_ubvi, 75, axis=0), line_width=6.5, color=pal[0], line_dash='dashed')
fig3.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 50, axis=0), line_width=6.5, color=pal[1])#, legend='BBVI-70/(n+1)')
fig3.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 25, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig3.line(np.arange(rkl_bbvi.shape[1])+1, np.percentile(rkl_bbvi, 75, axis=0), line_width=6.5, color=pal[1], line_dash='dashed')
fig3.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 50, axis=0), line_width=6.5, color=pal[2])#, legend='BBVI-1/(n+1)')
fig3.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 25, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')
fig3.line(np.arange(rkl_bbvi2.shape[1])+1, np.percentile(rkl_bbvi2, 75, axis=0), line_width=6.5, color=pal[2], line_dash='dashed')


#plot the KL vs cput
fig4 = bkp.figure(width=1000, height=500,x_axis_label='CPU Time (s)', y_axis_label='KL(q || p)', y_axis_type='log')
preprocess_plot(fig4, '42pt', True)
for cput, kl, nm, clrid in [(cput_ubvi, rkl_ubvi, 'UBVI', 0), (cput_bbvi, rkl_bbvi, 'BBVI-70/(n+1)', 1), (cput_bbvi2, rkl_bbvi2, 'BBVI-1/(n+1)', 2)]:
  cput_25 = np.percentile(np.cumsum(cput, axis=1), 25, axis=0)
  cput_50 = np.percentile(np.cumsum(cput, axis=1), 50, axis=0)
  cput_75 = np.percentile(np.cumsum(cput, axis=1), 75, axis=0)
  rkl_25 = np.percentile(kl, 25, axis=0)
  rkl_50 = np.percentile(kl, 50, axis=0)
  rkl_75 = np.percentile(kl, 75, axis=0)
  fig4.circle(cput_50, rkl_50, color=pal[clrid], size=10)#, legend=nm)
  fig4.segment(x0=cput_50, y0=rkl_25, x1=cput_50, y1=rkl_75, color=pal[clrid], line_width=4)#, legend=nm)
  fig4.segment(x0=cput_25, y0=rkl_50, x1=cput_75, y1=rkl_50, color=pal[clrid], line_width=4)#, legend=nm)


postprocess_plot(fig, '42pt')
postprocess_plot(fig2, '42pt')
postprocess_plot(fig3, '42pt')
postprocess_plot(fig4, '42pt')

bkp.show(bkl.gridplot([[fig, fig2, fig3, fig4]]))

