import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
import bokeh.palettes 
from bokeh.models import Label

def proj(v, view_dir, view_x, view_y):
  p = v - v.dot(view_dir)*view_dir
  return np.array([p.dot(view_x), p.dot(view_y)])

def plot_sphere_grid(occluded):
  for psi in np.linspace(0, 2*np.pi, 30):
    phis = np.linspace(-np.pi/2., np.pi/2., 100)
    line = np.vstack((np.cos(psi)*np.cos(phis), np.sin(psi)*np.cos(phis), np.sin(phis))).T
    projL = line - line.dot(view_dir)[:, np.newaxis]*view_dir
    lx = projL.dot(view_x)
    ly = projL.dot(view_y)
    idx = line.dot(view_dir) <= 5e-2 if not occluded else line.dot(view_dir) > 0
    fig.line(lx[idx], ly[idx], color='RoyalBlue', alpha=0.2 if occluded else 1.)
  
  for phi in np.linspace(-np.pi/2., np.pi/2., 15):
    psis = np.linspace(-np.pi, np.pi, 100)
    line = np.vstack((np.cos(psis)*np.cos(phi), np.sin(psis)*np.cos(phi), np.ones(100)*np.sin(phi))).T
    projL = line - line.dot(view_dir)[:, np.newaxis]*view_dir
    lx = projL.dot(view_x)
    ly = projL.dot(view_y)
    idx = line.dot(view_dir) <= 5e-2 if not occluded else line.dot(view_dir) > 0
    fig.line(lx[idx], ly[idx], color='RoyalBlue', alpha=0.2 if occluded else 1.)


def plot_geodesic(fig, x, y, occluded, clr, view_dir, view_x, view_y, arrowed=False, a_factor=1., line_dash='dashed'):
  #get geodesic pts
  lmbs = np.linspace(0., 1., 100)
  g = lmbs[:, np.newaxis]*x + (1. - lmbs[:, np.newaxis])*y
  g /= np.sqrt((g**2).sum(axis=1))[:, np.newaxis]
  #get their projections into view coords
  p = np.zeros((g.shape[0], 2))
  for i in range(g.shape[0]):
    p[i, :] = proj(g[i, :], view_dir, view_x, view_y)
  #get those that are occluded
  t_shift_back=12
  line_dash_length = 20
  if occluded:
    idcs = g.dot(view_dir) > 0
    fig.line(p[idcs, 0], p[idcs, 1], line_width=10, line_color=clr, line_dash=[line_dash_length, line_dash_length] if line_dash == 'dashed' else line_dash, alpha=0.3*a_factor)
    if y.dot(view_dir) > 0 and arrowed:
      ang = np.arctan2( p[idcs, 1][-1] - p[idcs, 1][-2], p[idcs, 0][-1] - p[idcs, 0][-2] )
      fig.triangle(p[idcs, 0][-t_shift_back], p[idcs, 1][-t_shift_back], size=40, angle=(ang-np.pi/2.), line_color=None, fill_color=clr, alpha=0.3*a_factor) 
  else:
    idcs = g.dot(view_dir) <= 0
    fig.line(p[idcs, 0], p[idcs, 1], line_width=10, line_color=clr, line_dash=[line_dash_length, line_dash_length] if line_dash == 'dashed' else line_dash, alpha=a_factor)
    if y.dot(view_dir) <= 0 and arrowed:
      ang = np.arctan2( p[idcs, 1][-1] - p[idcs, 1][-2], p[idcs, 0][-1] - p[idcs, 0][-2] )
      fig.triangle(p[idcs, 0][-t_shift_back], p[idcs, 1][-t_shift_back], size=40, angle=(ang-np.pi/2.), line_color=None, fill_color=clr, alpha=a_factor)

def plot_point(fig, x, occluded, clr, view_dir, view_x, view_y, label=None, label_shift_x=0, label_shift_y=0, occluded_alpha=0.3, label_size='80pt'):
  p = proj(x, view_dir, view_x, view_y) 
  if occluded and x.dot(view_dir) > 0:
    fig.circle(p[0], p[1], size=40, fill_color=clr, alpha=occluded_alpha, line_color=None)
    if label is not None:
      lbl = Label(x=p[0]+label_shift_x, y=p[1]+label_shift_y, text=label, render_mode='css', text_font_size=label_size, text_color=clr,text_alpha=occluded_alpha,
          border_line_color=None,
          background_fill_color=None, background_fill_alpha=0.0)
      fig.add_layout(lbl)
  elif (not occluded) and x.dot(view_dir) <= 0:
    fig.circle(p[0], p[1], size=40, fill_color=clr, line_color=None)
    if label is not None:
      lbl = Label(x=p[0]+label_shift_x, y=p[1]+label_shift_y, text=label, render_mode='css', text_font_size=label_size,text_color=clr,
          border_line_color=None,
          background_fill_color=None, background_fill_alpha=0.0)
      fig.add_layout(lbl)

def plot_vec(fig, x, y, occluded, clr, nrm, view_dir, view_x, view_y):
  xy = y - y.dot(x)*x
  xy /= np.sqrt((xy**2).sum())
  pxy = proj(xy, view_dir, view_x, view_y)
  px = proj(x, view_dir, view_x, view_y)
  fig.segment(px[0], px[1], px[0]+nrm*pxy[0], px[1]+nrm*pxy[1], line_color=clr, line_width=8)
  ang = np.arctan2(pxy[1], pxy[0])
  fig.triangle(px[0]+nrm*pxy[0], px[1]+nrm*pxy[1], size=40, angle=(ang-np.pi/2.), line_color=None, fill_color=clr)
 

#flags
plot_gbar = True
plot_f = True
plot_h = True
plot_gn1 = True
plot_gstar = True

pal = bokeh.palettes.colorblind['Colorblind'][8]
pal = [pal[0], pal[1], pal[3], pal[4], pal[5], pal[6], pal[7], pal[2]]

fig = bkp.figure(width=1000, height=1000)
fig.xgrid.grid_line_color=None
fig.ygrid.grid_line_color=None

#compute viewing frame
view_dir = -np.ones(3)/np.sqrt(3)
view_x = np.zeros(3)
view_x[0] = -1./np.sqrt(2.)
view_x[1] = 1./np.sqrt(2.)
view_y = -np.ones(3)/np.sqrt(6.)
view_y[2] = 2./np.sqrt(6.)

#set the cur pt, goal pt, and candidate geodesic endpts
x = np.array([5., 1., 5.])
x /= np.sqrt((x**2).sum())

xw = np.array([0., 1., 1.])
xw /= np.sqrt((xw**2).sum())

xg = np.zeros((4, 3))
xg[0, :] = np.array([3., 1., -1.])
xg[1, :] = np.array([0., 1., 0.])
xg[2, :] = np.array([-0.5, -1., -0.1])
xg[3, :] = np.array([0.5, -2., 2.])
xg /= np.sqrt((xg**2).sum(axis=1))[:, np.newaxis]


xorth = xg[3,:] - xg[3,:].dot(xw)*xw
xorth = xorth/np.sqrt((xorth**2).sum())

xstar = np.sqrt( x.dot(xorth) / (x.dot(xorth) + x.dot(xw)))
xx = xstar*xorth + np.sqrt(1.-xstar**2)*xw
px = proj(xx, view_dir, view_x, view_y) 

for occluded in [True, False]:
  if plot_f:
    plot_point(fig, x, occluded, pal[0], view_dir, view_x, view_y, label='f', label_shift_x=-0.15, label_shift_y=-0.15)
    plot_geodesic(fig, xw, x, occluded, pal[0], view_dir, view_x, view_y)
    plot_vec(fig, xw, x, occluded, pal[0], 0.35, view_dir, view_x, view_y)
  if plot_h:
    for i in range(3):
      plot_point(fig, xg[i, :], occluded, 'black', view_dir, view_x, view_y, label='h', label_shift_x=-0.2, label_shift_y=-0.0)
      plot_geodesic(fig, xw, xg[i, :], occluded, 'black', view_dir, view_x, view_y)
      plot_vec(fig, xw, xg[i, :], occluded, 'black', 0.35, view_dir, view_x, view_y)
  if plot_gn1:
    plot_point(fig, xg[3, :], occluded, pal[1], view_dir, view_x, view_y, label='g', label_shift_x=-0.2, label_shift_y=0.1)
    plot_point(fig, xg[3, :], occluded, pal[1], view_dir, view_x, view_y, label='n+1', label_shift_x=-0.07, label_shift_y=0.1, label_size='56pt')
    plot_geodesic(fig, xw, xg[3, :], occluded, pal[1], view_dir, view_x, view_y)
    plot_vec(fig, xw, xg[3,:], occluded, pal[1], 0.35, view_dir, view_x, view_y)
  if plot_gstar: 
    plot_point(fig, xx, occluded, pal[1], view_dir, view_x, view_y, label='g*', label_shift_x=-0.1, label_shift_y=-0.3)
    plot_geodesic(fig, xw, xx, occluded, pal[1], view_dir, view_x, view_y, line_dash='solid')
  #lbl = Label(x=px[0], y=px[1] -0.4, text='x*', render_mode='css', text_font_size='80pt', text_color=pal[1],
  #        border_line_color=None,
  #        background_fill_color=None, background_fill_alpha=0.0)
  #fig.add_layout(lbl)


  if plot_gbar:
    plot_point(fig, xw, occluded, 'black', view_dir, view_x, view_y, label='g\u0305\u2099',label_shift_x=0.15, label_shift_y=-0.1)

  plot_sphere_grid(occluded)
    
  if occluded:
    fig.patch(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), fill_color='RoyalBlue', alpha=0.1) 

#only plot xg[0, :] here since this is the greedy choice of geodesic
g0 = x.dot(xg[0, :])
g1 = x.dot(xw)
g2 = xw.dot(xg[0, :])
gamma = (g0-g1*g2)/(g0-g1*g2 + g1-g0*g2)
y = xw + gamma*(xg[0, :] - xw)
y /= np.sqrt((y**2).sum())

vy = xw - y.dot(xw)*y 
vy /= -np.sqrt((vy**2).sum())


fig2 = bkp.figure(width=1000, height=1000)
fig2.xgrid.grid_line_color=None
fig2.ygrid.grid_line_color=None

#plot occluded stuff
plot_point(fig2, x, True, pal[0], view_dir, view_x, view_y)
plot_point(fig2, xg[0, :], True, 'black', view_dir, view_x, view_y)
plot_geodesic(fig2, xw, xg[0, :], True, 'black', view_dir, view_x, view_y, a_factor=0.5)
for i in range(1, xg.shape[0]):
  plot_point(fig2, xg[i, :], True, 'black', view_dir, view_x, view_y)
plot_geodesic(fig2, y, xw, True, pal[1], view_dir, view_x, view_y, arrowed=True)
plot_geodesic(fig2, y, x, True, pal[0], view_dir, view_x, view_y)
plot_point(fig2, y, True, pal[1], view_dir, view_x, view_y)


#plot sphere grid
for psi in np.linspace(0, 2*np.pi, 30):
  phis = np.linspace(-np.pi/2., np.pi/2., 100)
  line = np.vstack((np.cos(psi)*np.cos(phis), np.sin(psi)*np.cos(phis), np.sin(phis))).T
  projL = line - line.dot(view_dir)[:, np.newaxis]*view_dir
  lx = projL.dot(view_x)
  ly = projL.dot(view_y)
  idx = line.dot(view_dir) <= 5e-2
  fig2.line(lx[idx], ly[idx], color='RoyalBlue')
  idx = line.dot(view_dir) > 0
  fig2.line(lx[idx], ly[idx], alpha=0.2, color='RoyalBlue')

for phi in np.linspace(-np.pi/2., np.pi/2., 15):
  psis = np.linspace(-np.pi, np.pi, 100)
  line = np.vstack((np.cos(psis)*np.cos(phi), np.sin(psis)*np.cos(phi), np.ones(100)*np.sin(phi))).T
  projL = line - line.dot(view_dir)[:, np.newaxis]*view_dir
  lx = projL.dot(view_x)
  ly = projL.dot(view_y)
  idx = line.dot(view_dir) <= 5e-2
  fig2.line(lx[idx], ly[idx], color='RoyalBlue')
  idx = line.dot(view_dir) > 0
  fig2.line(lx[idx], ly[idx], alpha=0.2, color='RoyalBlue')

#color the sphere
fig2.patch(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), fill_color='RoyalBlue', alpha=0.1) 

#plot non occluded stuff
plot_point(fig2, x, False, pal[0], view_dir, view_x, view_y)
plot_point(fig2, xg[0, :], False, 'black', view_dir, view_x, view_y)
plot_geodesic(fig2, xw, xg[0, :], False, 'black', view_dir, view_x, view_y, a_factor=0.5)
for i in range(1, xg.shape[0]):
  plot_point(fig2, xg[i, :], False, 'black', view_dir, view_x, view_y)
plot_geodesic(fig2, y, xw, False, pal[1], view_dir, view_x, view_y, arrowed=True)
plot_geodesic(fig2, y, x, False, pal[0], view_dir, view_x, view_y)
plot_point(fig2, y, False, pal[1], view_dir, view_x, view_y)


bkp.show(bkl.gridplot([[fig, fig2]]))

