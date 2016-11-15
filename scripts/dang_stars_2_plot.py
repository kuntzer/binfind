import numpy as np
from matplotlib import pyplot as plt
import itertools
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
	figures.set_fancy(txtsize=18)
	
reload_data = False

e_req = 0.01#2e-4
r_req = 0.05#1e-3

n_angsep = 15
n_con = 15
# Minimum separation of the stars to be qualified as binaries
crits_angsep = np.linspace(1, 15, n_angsep)
# Max contrast to be qualified as binaries
crits_contrast = np.linspace(0.1, 1.5, n_con)#np.linspace(0.1, 0.015, 7)

save = True
outdir = 'data/binfind_percent_meas/dang_stars'

e1_deforms = []
e2_deforms = []
r2_deforms = []

if reload_data:
	criteria = list(itertools.product(*[crits_angsep, crits_contrast]))
	for iix, (crit_angsep, crit_contrast) in enumerate(criteria):
		
		ca, cc, e1_deform, e2_deform, r2_deform = u.readpickle(os.path.join(outdir, 'dang_stars_{:d}_{:1.1f}.pkl'.format(int(crit_angsep), crit_contrast)))
		e1_deforms.append(np.percentile(e1_deform, [95])[0])
		e2_deforms.append(np.percentile(e2_deform, [95])[0])
		r2_deforms.append(np.percentile(r2_deform, [95])[0])
		print iix, crit_angsep, crit_contrast, e1_deforms[-1], e2_deforms[-1], r2_deforms[-1]
		#
	
	e1_deforms = np.asarray(e1_deforms)
	e2_deforms = np.asarray(e2_deforms)
	r2_deforms = np.asarray(r2_deforms)
	
	e1_deform = e1_deforms.reshape([n_angsep, n_con])
	e2_deform = e2_deforms.reshape([n_angsep, n_con])
	r2_deform = r2_deforms.reshape([n_angsep, n_con])
	
	u.writepickle([e1_deform, e2_deform, r2_deform], os.path.join(outdir, 'resume_dang_stars.pkl'))
else:
	e1_deform, e2_deform, r2_deform = u.readpickle(os.path.join(outdir, 'resume_dang_stars.pkl'))


# Let's start by constructing the meshgrid and then the data variable
x = crits_angsep 
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

CS = plt.pcolormesh(X, Y, e1_deform.T, cmap=plt.get_cmap("inferno_r"))#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])
cbar = plt.colorbar(CS)#, ticks=np.linspace(0,1,21))
cbar.set_label(r"$\Delta e_1$")
plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	
fig2 = plt.figure()
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
CS = plt.pcolormesh(X, Y, e2_deform.T, cmap=plt.get_cmap("inferno_r"))#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])
cbar = plt.colorbar(CS)#, ticks=np.linspace(0,1,21))
cbar.set_label(r"$\Delta e_2$")
plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])

fig3 = plt.figure()
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
CS = plt.pcolormesh(X, Y, r2_deform.T, cmap=plt.get_cmap("inferno_r"))#, vmin=vmin, vmax=vmax)
#plt.axis([X.min(),X.max(),Y.min(),Y.max()])
#plt.xticks(x[:-1][::2])
#plt.yticks(y[:-1])
cbar = plt.colorbar(CS)#, ticks=np.linspace(0,1,21))
cbar.set_label(r"$\Delta R^2/R^2$")
plt.xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
plt.ylabel(r"$\mathrm{Contrast\ [mag]}$")
plt.axis([X.min(),X.max(),Y.min(),Y.max()])


fig4 = plt.figure(figsize=(8.8,7.2))
ax = plt.subplot(111)
# Round up to the nearest 0.05
#vmin = np.round(Zd.min() * 20) / 20
#vmax = np.round(Zd.max() * 20) / 20
mean_budg = (r2_deform.T / r_req + (e1_deform + e2_deform).T / e_req)/3. * 100
mean_budg = ((e1_deform + e2_deform).T)/2. * 1e2
r_tab = r2_deform.T * 1e2

CS1 = plt.pcolormesh(X, Y, r2_deform.T / r_req, cmap=plt.get_cmap("inferno_r"))

vmaxe = np.round(mean_budg.max() / .10, 0) * .10
vmine = np.round(mean_budg.min() / .10, 0) * .10

tt = 1e3
vmaxr = np.round(r_tab.max() * tt, 0) / tt
vminr = np.floor( np.round(r_tab.min() * tt, 0) / tt * 1e2) / 1e2
print vmine
print vminr, vmaxr
print r2_deform.min(), r2_deform.max() 

#CS1 = plt.pcolormesh(X, Y, r_tab, cmap=plt.get_cmap("inferno_r"), vmin=vminr, vmax=vmaxr)
CS = plt.pcolormesh(X, Y, mean_budg, cmap=plt.get_cmap("inferno_r"), vmin=vmine, vmax=vmaxe)
plt.xticks(x[:-1][::2])
plt.yticks(y[:-1][::2])

import matplotlib.ticker as ticker

tickse = np.arange(vmine, vmaxe, .5)
ticksr = np.linspace(vminr, vmaxr, 5) 
cbar = plt.colorbar(CS, ticks=tickse, pad=0.01)

if white_plot:
	commonticks = ["%1.1f%; %1.2f%$" % (tickse[ii], ticksr[ii]) for ii in range(len(tickse))]
else:
	commonticks = [r"$\smallskip%1.1f\%%$" % tickse[ii] + "\n" + r"$%1.2f\%%$" % (ticksr[ii]) for ii in range(len(tickse))]

cbar.ax.set_yticklabels(commonticks, ha = 'left')
if white_plot:
	cbar.set_label("$\Delta$e/e_0;\,\Delta R^2/R_0^2$", color="white")
	plt.xlabel("Maximum angular separation [mas]")
	plt.ylabel("Minimum contrast [mag]")
else:
	cbar.set_label(r"$\langle\Delta e_i/e_{0,i}\rangle;\,\langle\Delta R^2/R^2_{0}\rangle$")
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
