import numpy as np
from matplotlib import pyplot as plt
import bf_utils as bf
import os 
import figures

white_plot = False

if white_plot:
	# Include this for presentations:
	import matplotlib.font_manager as fm
	from matplotlib import rc
	prop = fm.FontProperties(fname='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreadventor-regular.otf')
	#rc('font', **{'fname':'/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreadventor-regular.otf'})
	rc('font', **{'family':'TeX Gyre Adventor','size':14})
#### End
else:
	figures.set_fancy()

e_req = 0.01#2e-4
r_req = 0.05#1e-3

# Start from zero
computes = False

save = True

outdir = 'data/binfind_percent_meas/dang_stars'
crits_angsep, crits_contrast, e1_deform, e2_deform, r2_deform = bf.readpickle(os.path.join(outdir, 'dang_stars.pkl'))

# Let's start by constructing the meshgrid and then the data variable
x = crits_angsep * 1e3
y = crits_contrast

dx = (x[1]-x[0])/2.
dy = (y[1]-y[0])/2.

x = np.hstack([x, x[-1] + 2.*dx])
y = np.hstack([y, y[-1] + 2.*dy])
X, Y = np.meshgrid(x-dx,y-dy)
	
fig1 = plt.figure()
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
CS = plt.pcolormesh(X, Y, e1_deform.T * 1e4, cmap=plt.get_cmap("inferno_r"))#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])
cbar = plt.colorbar(CS)#, ticks=np.linspace(0,1,21))
cbar.set_label(r"$10^4\cdot\Delta e_1$")
plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	
fig2 = plt.figure()
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
CS = plt.pcolormesh(X, Y, e2_deform.T * 1e4, cmap=plt.get_cmap("inferno_r"))#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])
cbar = plt.colorbar(CS)#, ticks=np.linspace(0,1,21))
cbar.set_label(r"$10^4\cdot\Delta e_2$")
plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])

fig3 = plt.figure()
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
CS = plt.pcolormesh(X, Y, r2_deform.T * 1e3, cmap=plt.get_cmap("inferno_r"))#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])
cbar = plt.colorbar(CS)#, ticks=np.linspace(0,1,21))
cbar.set_label(r"$10^3\cdot\Delta R^2/R^2$")
plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])


fig4 = plt.figure()
ax = plt.subplot(111)
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
mean_budg = (r2_deform.T / r_req + (e1_deform + e2_deform).T / e_req)/3. * 100

vmax = np.round(mean_budg.max() / 1, 0) * 1
CS = plt.pcolormesh(X, Y, mean_budg, cmap=plt.get_cmap("inferno_r"), vmax=vmax)#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])

import matplotlib.ticker as ticker

def fmt(x, pos):
	if white_plot:
		return '%1.1f%%' % x
	else:
		return r'$%1.1f\%%$' % x

print mean_budg.max() 

if vmax < 10:
	ticks = np.arange(0,vmax+0.5,0.5)
else:
	ticks = np.linspace(0,vmax,11)
cbar = plt.colorbar(CS, format=ticker.FuncFormatter(fmt), ticks=ticks)#, ticks=np.linspace(0,1,21))
if white_plot:
	cbar.set_label("Mean  Error/Measurement error", color="white")
	plt.xlabel("Maximum angular separation [mas]")
	plt.ylabel("Minimum contrast [mag]")
else:
	cbar.set_label(r"$\mathrm{Mean}\quad \mathrm{Error}/\mathrm{Measurement\ error}$")
	plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
	plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])

if white_plot:
	[ ax.spines[s].set_color('white') for s in ax.spines]
	ax.xaxis.label.set_color('white')
	ax.tick_params(axis='x', colors='white')
	ax.yaxis.label.set_color('white')
	ax.tick_params(axis='y', colors='white')
	#cbar.outline.set_color('w')                   #set colorbar box color
	cbar.ax.yaxis.set_tick_params(color='w')      #set colorbar ticks color 
	cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')                #tricky
	
	plt.setp(cbytick_obj, color='w')
	#cbar.outline.set_color('w') 

if save:
	figures.savefig(os.path.join(outdir, "dang_e1"), fig1, fancy=True, pdf_transparence=True)
	figures.savefig(os.path.join(outdir, "dang_e2"), fig2, fancy=True, pdf_transparence=True)
	figures.savefig(os.path.join(outdir, "dang_r2"), fig3, fancy=True, pdf_transparence=True)
	figures.savefig(os.path.join(outdir, "dang_summary"), fig4, fancy=True, pdf_transparence=True)

plt.show()
