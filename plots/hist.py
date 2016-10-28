from __future__ import division

import numpy as np
import pylab as plt
from scipy import stats

def hist(ax, stars_characteristics, predictions):
	"""
	
	"""
	
	binary_stars = stars_characteristics[:,0]
	idbin = np.where(binary_stars == 1)
	
	all_stars = stars_characteristics[idbin, 1].flatten() / .12 # Convert from Euclid small px to mas
	_, binst, _ = ax.hist(all_stars, 100, histtype='stepfilled', color="r", alpha=0.8)
		
	observed = np.where(np.logical_and(binary_stars == 1, predictions))
	obs_stars = stars_characteristics[observed, 1].flatten() / .12 # Convert from Euclid small px to mas
	ax.hist(obs_stars, binst, histtype='stepfilled', color="darkgrey", alpha=0.8)
	
	binst = stats.mstats.mquantiles(all_stars, [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.])
	#binst = 6
	nt, binst = np.histogram(all_stars, binst)
	nd, _ = np.histogram(obs_stars, bins=binst)
	xbin = [(binst[i] + binst[i+1])/2. for i in range(len(binst) - 1)]
	ybin = nd/nt
	#xbin = np.concatenate([[all_stars.min()], xbin])
	#ybin = np.concatenate([[ybin[0]], ybin])
	
	ax2 = ax.twinx()
	ax2.set_ylabel(r'$\mathrm{Completeness}$')
	ax2.plot(xbin, ybin, c='k', lw=2)
	ax2.set_ylim([0, 1])
	
	plt.setp(ax.get_yticklabels(), visible=False)
		
	ax2.grid(True)
	ax.set_xlabel(r"$\mathrm{Angular\ separation\ [mas]}$")
	ax.set_ylabel(r"$\mathrm{Occurence\ of\ binaries\ (arbit.\ units)}$")
	
	ax.set_xlim([all_stars.min(), (all_stars.max())])
	