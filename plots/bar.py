
def errorbar(ax, y, yerr, barlabels, convertlabels=True, **kwargs):
	sizedata = len(barlabels)
	
	mykwargs = {"alpha":1., "lw":1.5, "color":'goldenrod', "ecolor":'k', 'error_kw':{'lw':1.5}}
	# We overwrite these mykwargs with any user-specified kwargs:
	mykwargs.update(kwargs)
		
	ax.bar(range(sizedata), y,
       yerr=yerr, align="center", **mykwargs)

	ax.set_xticks(range(sizedata))
	if convertlabels:
		barlabels = [r"${:d}$".format(l) for l in barlabels]
	ax.set_xticklabels(barlabels)
	
	ax.set_xlim([-1, sizedata])
	ax.set_ylabel(r"$\mathrm{Feature\ importance}$")