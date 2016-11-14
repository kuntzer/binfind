import pylab as plt
import numpy as np
import os

import binfind.utils as u
import binfind.plots.figures as figures
figures.set_fancy(20)

indir = "data/binfind_percent_meas/ukn_PSF"

results_train, results_test = u.readpickle(os.path.join(indir, "results.pkl"))

axes = {"F1_min":10, "AUC_min":10, "F1_max":-10, "AUC_max":-10}

print 'INDIR is', indir
def r(value):
	return np.floor(value.min() * 10) / 10

for key in results_test:
	print 'TREATING RESULT for', key
	data = results_test[key]
	
	# A bit of a pain in the A** this pcolormesh
	# Let's start by constructing the meshgrid and then the data variable
	x = np.unique(data[:,0])
	y = np.unique(data[:,1])
	
	x *= 1e3 # Let's display in milli-arcsec
	dx = (x[1]-x[0])/2.
	dy = (y[1]-y[0])/2.
	
	x = np.hstack([x, x[-1] + 2.*dx])
	y = np.hstack([y, y[-1] + 2.*dy])
	X, Y = np.meshgrid(x-dx,y-dy)
	
	axes["{:s}_X".format(key)] = X
	axes["{:s}_Y".format(key)] = Y

	F1 = (data[:,5].reshape([np.unique(data[:,0]).size, np.unique(data[:,1]).size])).T
	Zd = (data[:,9].reshape([np.unique(data[:,0]).size, np.unique(data[:,1]).size])).T
	FPR = (data[:,4].reshape([np.unique(data[:,0]).size, np.unique(data[:,1]).size])).T
	
	axes["{:s}_F1".format(key)] = F1
	axes["{:s}_AUC".format(key)] = Zd
	
	if r(F1.min()) < axes["F1_min"]:
		axes["F1_min"] = r(F1.min())
	if r(Zd.min()) < axes["AUC_min"]:
		axes["AUC_min"] = r(Zd.min())
		
	if r(F1.max()) > axes["F1_max"]:
		axes["F1_max"] = r(F1.max())
	if r(Zd.max()) > axes["AUC_max"]:
		axes["AUC_max"] = r(Zd.max())
	
	print 'thr variations:'
	usedthr = (data[:,2].reshape([np.unique(data[:,0]).size, np.unique(data[:,1]).size])).T
	for line in usedthr:
		for l in line:
			print '{:1.3f}'.format(l),
		print
	print "mean/median thr:", np.mean(usedthr), np.median(usedthr)

	plt.figure()
	CS = plt.pcolormesh(X, Y, FPR, cmap=plt.get_cmap("viridis"))#, vmin=vmin, vmax=vmax)
	plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	plt.xticks(x[:-1][::2])
	plt.yticks(y[:-1][::2])
	cbar = plt.colorbar(CS)
	cbar.set_label(r"$\mathrm{FPR}$")
	plt.xlabel(r"$\mathrm{Minimum\ angular\ separation\ [mas]}$")
	plt.ylabel(r"$\mathrm{Maximum\ contrast\ [mag]}$")
	plt.show()
	continue
	# Plot the F1 score using pcolormesh
	fig1 = plt.figure(figsize=(6.5,5.))
	plt.subplots_adjust(wspace=0.01)
	plt.subplots_adjust(bottom=0.19)
	plt.subplots_adjust(top=0.96)
	plt.subplots_adjust(left=0.14)
	plt.subplots_adjust(right=0.94)
	# Round up to the nearest 0.1
	vmin = np.floor(F1.min() * 10) / 10
	vmax = np.ceil(F1.max() * 10) / 10
	CS = plt.pcolormesh(X, Y, F1, cmap=plt.get_cmap("plasma"), vmin=vmin, vmax=vmax)
	plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	plt.xticks(x[:-1][::2])
	plt.yticks(y[:-1][::2])
	cbar = plt.colorbar(CS, ticks=np.linspace(0,1,11))
	cbar.set_label(r"$\max\ F_1\mathrm{\ score}$")
	plt.xlabel(r"$\mathrm{Minimum\ angular\ separation\ [mas]}$")
	plt.ylabel(r"$\mathrm{Maximum\ contrast\ [mag]}$")
	
	# This is for the AUC score
	fig2 = plt.figure()
	# Round up to the nearest 0.05
	vmin = 0.5#np.round(Zd.min() * 20) / 20
	vmax = 0.85#np.round(Zd.max() * 20) / 20
	CS = plt.pcolormesh(X, Y, Zd, cmap=plt.get_cmap("viridis"), vmin=vmin, vmax=vmax)
	plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	plt.xticks(x[:-1][::2])
	plt.yticks(y[:-1][::2])
	cbar = plt.colorbar(CS, ticks=np.linspace(0,1,21))
	cbar.set_label(r"$\mathrm{AUC\ score}$")
	plt.xlabel(r"$\mathrm{Minimum\ angular\ separation\ [mas]}$")
	plt.ylabel(r"$\mathrm{Maximum\ contrast\ [mag]}$")
	
	
	# This is the two plots combined
	fig3 = plt.figure(figsize=(10,5.))
	plt.subplots_adjust(wspace=0.01)
	plt.subplots_adjust(bottom=0.14)
	plt.subplots_adjust(top=0.98)
	ax1 = plt.subplot(121)
	# Round up to the nearest 0.05
	vmin = np.floor(F1.min() * 10) / 10
	vmax = np.ceil(F1.max() * 10) / 10
	CS = plt.pcolormesh(X, Y, F1, cmap=plt.get_cmap("plasma"), vmin=vmin, vmax=vmax)
	plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	plt.xticks(x[:-1][::2])
	plt.yticks(y[:-1][::2])
	plt.xlabel(r"$\mathrm{Minimum\ angular\ separation\ [mas]}$")
	plt.ylabel(r"$\mathrm{Maximum\ contrast\ [mag]}$")
	
	cbar = plt.colorbar(CS, ticks=np.linspace(0,1,11), orientation='horizontal',fraction=0.035)#, pad=0.04)
	cbar.set_label(r"$\max\ F_1\mathrm{\ score}$")
	
	
	ax2 = plt.subplot(122)
	# Round up to the nearest 0.1
	vmin = np.floor(Zd.min() * 10) / 10
	vmax = np.ceil(Zd.max() * 10) / 10
	CS = plt.pcolormesh(X, Y, Zd, cmap=plt.get_cmap("viridis"), vmin=vmin, vmax=vmax)
	plt.axis([X.min(),X.max(),Y.min(),Y.max()])
	plt.xticks(x[:-1][::2])
	plt.yticks(y[:-1][::2])
	ax2.set_yticklabels([])
	cbar = plt.colorbar(CS, ticks=np.linspace(0,1,11), orientation='horizontal',fraction=0.035)
	cbar.set_label(r"$\mathrm{AUC\ score}$")
	#plt.xlabel(r"$\mathrm{Minimum\ angular\ separation\ [mas]}$")
	#plt.ylabel(r"$\mathrm{Maximum\ contrast\ [mag]}$")

	fname1 = os.path.join(indir, 'figures', 'results_{:s}_f1_alpha_contr'.format(key))
	fname2 = os.path.join(indir, 'figures', 'results_{:s}_auc_alpha_contr'.format(key))	
	fname3 = os.path.join(indir, 'figures', 'results_{:s}_f1auc_alpha_contr'.format(key))		
	print 'Saving plots'
	figures.savefig(fname1, fig1, fancy=True, pdf_transparence=True)
	figures.savefig(fname2, fig2, fancy=True, pdf_transparence=True)
	figures.savefig(fname3, fig3, fancy=True, pdf_transparence=True)
	
	plt.show()
	
# Let's do the common plots


def set_ax(ax_, x, y):
	ax_.set_xticks(x[:-1][::2])
	ax_.set_yticks(y[:-1][::2])

keys = sorted(results_test.keys())
lenk = len(results_test)
fig4, plax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
plt.subplots_adjust(wspace=0.01)
plt.subplots_adjust(hspace=0.01)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(right=0.98)
plt.subplots_adjust(top=0.98)

box_props = dict(boxstyle="round", fc="w", ec="0.3", alpha=0.3)

axes["F1_max"] = 1
print axes["AUC_max"]
axes["AUC_max"] = 0.95

for ii, ax_ in enumerate(plax.flat):
	
	keyid = ii / lenk
	
	key = keys[keyid]
	
	if ii % lenk == 0:
		D = axes["{:s}_F1".format(key)]
		cmap = plt.get_cmap("plasma")
		ax_.annotate(r'$\mathrm{%s}$' % key.upper(), xy=(0.1, 0.8), xycoords="axes fraction", bbox=box_props)
		vmin = axes["F1_min"]
		vmax = axes["F1_max"]
	elif ii % lenk == 1:
		D = axes["{:s}_AUC".format(key)]
		cmap = plt.get_cmap("viridis")
		ax_.set_yticklabels([])
		vmin = axes["AUC_min"]
		vmax = axes["AUC_max"]
	else:
		raise NotImplemented()
	
	X = axes["{:s}_X".format(key)]
	Y = axes["{:s}_Y".format(key)]
	
	im = ax_.pcolormesh(X, Y, D, cmap=cmap, vmin=vmin, vmax=vmax)
	
	if ii % lenk == 0:
		f1_CS = im
	elif ii % lenk == 1:
		auc_CS = im
		
	ax_.set_xticks(x[:-1][::2])
	ax_.set_yticks(y[:-1][::2])
	ax_.set_xlim([X.min(), X.max()])
	ax_.set_ylim([Y.min(), Y.max()])
fig4.text(0.575, 0.18, r"$\mathrm{Minimum\ angular\ separation\ [mas]}$", ha='center')
fig4.text(0.05, 0.625, r"$\mathrm{Maximum\ contrast\ [mag]}$", va='center', rotation='vertical')

position=fig4.add_axes([0.21,0.11,0.3,0.05])
cbar = fig4.colorbar(f1_CS, cax=position, ticks=np.linspace(0,1,6), orientation='horizontal')
cbar.set_label(r"$F_1\mathrm{\ score}$")
position=fig4.add_axes([0.61,0.11,0.32,0.05])
cbar = fig4.colorbar(auc_CS, cax=position, ticks=np.linspace(0,1,11), orientation='horizontal')
cbar.set_label(r"$\mathrm{AUC\ score}$")

fname4 = os.path.join(indir, 'figures', 'results_all_f1auc_alpha_contr'.format(key))	
figures.savefig(fname4, fig4, fancy=True, pdf_transparence=True)

plt.show()
	
	