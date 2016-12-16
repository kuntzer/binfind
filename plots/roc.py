import numpy as np
import pylab as plt
from matplotlib.collections import LineCollection

from .. import diagnostics
import figures

def roc(ax, params, metrics=None, metrics_label=None, colors=None, labels=None, id_fpr=2, id_tpr=1, **kwargs):
	"""
	Makes a ROC plot for one or several classifier on an existing axis. 
	
	:param ax: the ax instance to plot on
	:param params: a *list* of parameters to extract the FPR and TPR
	:param show_metrics: which metrics to show in addition to FPR, default: None
	:param colors: A list of the color to use when plotting if None, uses default colors
	:param labels: A list of labels to use.
	
	Any further kwargs are either passed to ``plot()``.
	
	Some commonly used kwargs for plot() are:
	
	* **marker**: marker type
	* **ms**: marker size in points
	* **ls**: linestyle
	* **lw**: linewdith
	"""
	
	if colors is None:
		colors = figures.get_colors()
		
	if labels is None:
		labels = len(params) * [None]
		
	if metrics is None:
		metrics = len(params) * [None]

	"""
	import matplotlib
	#cmap = matplotlib.colors.Colormap('plasma')
	cmap = plt.cm.get_cmap('plasma')#, N=5)#, 11)#'plasma', 11)
	
	# define the colormap
	# extract all colors from the .jet map
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# force the first color entry to be grey
	#cmaplist[0] = (.5,.5,.5,1.0)
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	"""
	import matplotlib.colors

	
	# Round up to the nearest 0.1
	metrics_vs = np.array(metrics)
	metric_vmin = 0#np.floor(np.amin(metrics_vs) * 10) / 10
	metric_vmax = 1#np.ceil(np.amax(metrics_vs) * 10) / 10

	for p, c, l, metric in zip(params, colors, labels, metrics):
		
		tpr_, fpr_, indx = diagnostics.get_unique_tpr_fpr(p, id_fpr, id_tpr, return_indx=True)
		auc = diagnostics.auc(p)
		
		if l is None:
			l = r'$AUC={:.02f}$'.format(auc)
		else:
			l = r'${:s}={:.02f}$'.format(l, auc)
			
		mykwargs = {"ls":"-", "alpha":1., "lw":2.5, "color":c, "label":l}
		# We overwrite these mykwargs with any user-specified kwargs:
		mykwargs.update(kwargs)
			
		ax.plot(fpr_, tpr_, **mykwargs)
		
		if metric is not None:
			m = np.concatenate([[metric[-1]], metric[indx], [metric[0]]])
			
			cmap = plt.cm.get_cmap('plasma')
			bounds = np.linspace(0, 1, 6)
			norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
			
			cax = ax.scatter(fpr_, tpr_, edgecolor="None", c=m, cmap=cmap, s=90, marker="o", alpha=0.8, vmin=metric_vmin, vmax=metric_vmax, norm=norm)
			
			points = np.array([fpr_, tpr_]).T.reshape(-1, 1, 2)
			segments = np.concatenate([points[:-1,::1], points[1:,::1]], axis=1)
	
			
			lc = LineCollection(segments, cmap=cmap, alpha=0.8, norm=norm)
			lc.set_array(m)
			lc.set_linewidth(10)
			plt.gca().add_collection(lc)
			
	if metric is not None:
		cb = plt.colorbar(cax, ticks=np.linspace(0,1,6))
		if metrics_label is None:
			metrics_label = r"$\mathrm{Metrics\ score}$"
		cb.set_label(metrics_label)
			
	
	ax.plot([0, 1], [0, 1], ls='--', c='k')
	ax.set_xlabel(r"$\mathrm{False\ positive\ rate}$")
	ax.set_ylabel(r"$\mathrm{True\ positive\ rate}$")
	ax.legend(loc=4)
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
		
	ax.grid(True)