import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

import binfind
import utils as u

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

n_exposures = 4
mlparams = binfind.classifier.MLParams(name = "foo", 
		features = range(15),  labels = range(1))
toolparams = binfind.classifier.fannwrapper.FANNParams(name = "bar", hidden_nodes = [15,15,15],
        max_iterations = 2000)
binfind.plots.figures.set_fancy()


###################################################################################################
### PARAMETERS

## Simulation parameters
# Minimum separation of the stars to be qualified as binaries
crits_angsep = [0.014,0.015]#np.linspace(0.001, 0.015, 15)
# Max contrast to be qualified as binaries
crits_contrast = np.linspace(0.1, 1.5, 15)
# Number of times to do the whfname_interpolationole analysis 
n_training = 25
n_validation = 5
n_test = 15
# Number of stars per field 
n_stars = 280
# Bin fraction to reach
bin_fraction = 0.3
# Outdir
outdir = 'data/binfind_percent_meas/ukn_PSF_ann'

## Observables and quality parameters
# Stellar catalogue path and parameters
star_catalogues_path = '/home/kuntzer/workspace/Blending_PSF_Euclid/data/BGM/'
l, b = (180, 15)
# Path to interpolation file
fname_interpolation = 'data/measurements/interpolations.pkl'
# Path to fiducial position in x y of psf file
fname_fiducial = 'psf_fields/psf_ellip_gs.dat'
# Brightest magnitude observable
m_min = 18 
# faintest mag observable
m_max = 24.5

## Exposures parameters
# Number of stars per fields 
# What are the requirements on the reconstruction for a single star?
ei_max_error = 1e-2 # = 1% error
r2_max_error = 5e-2 # = 5% error

# What is the wanted False Positive Rate ? (in fraction)
thr_fpr = 0.1
recovery_n_inter = 2
recovery_n_neighbour = 10

# Thresholds for the star/multiple star classification
thresholds = np.logspace(-8, 0, 1000)

# Show figures after each criteria ?
show = False

###################################################################################################
### INITIALISATION

if len(crits_angsep) == 1 and len(crits_contrast) == 1:
	single_exp = True
else:
	f1_per_crit = []
	lim_per_crit = []
	single_exp = False
criteria = list(itertools.product(*[crits_angsep, crits_contrast]))

#data = blutil.load_bmg(os.path.join(star_catalogues_path, 'star_field_BGM_i_%d_%d_%d' % (l, b, fid)), main_sequence=True)
previous_sep = -1
data = None

psf_positions = np.loadtxt(fname_fiducial)
x_psf = psf_positions[:,0]
y_psf = psf_positions[:,1]
min_x_psf = np.amin(x_psf)
min_y_psf = np.amin(y_psf)
max_x_psf = np.amax(x_psf)
max_y_psf = np.amax(y_psf)

euclid = binfind.simulation.Observations(ei_max_error, r2_max_error, fname_interpolation, fname_fiducial)

for iix, (crit_angsep, crit_contrast) in enumerate(criteria):
	
	ml_class = binfind.classifier.ML(mlparams, toolparams)
	
	results_train = {'ann':[]}
	results_test = {'ann':[]}

	fids = u.get_id_catalogs(crit_angsep, crit_contrast)
			
	if len(fids) != previous_sep:	
		previous_sep = len(fids)
		data = None
		for fid in fids:
			fname = os.path.join(star_catalogues_path, 'star_field_BGM_i_%d_%d_%d' % (l, b, fid))
			datal = binfind.utils.load_bmg(fname, main_sequence=True)
			if data is None:
				data = datal
			else:
				data = np.vstack([data,datal]) 

	print "=" * 60
	print "Running experiments on alpha > %0.4f, contrast < %0.1f --- (%d/%d)" % (crit_angsep, crit_contrast, iix+1, len(criteria))

	sim_cat = binfind.simulation.Catalog(crit_angsep, crit_contrast)
	
	features = None
	

	###################################################################################################
	### CORE OF CODE
	"""
	for ith_experience in range(n_training):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_training)

		stars_to_observe, feature, fiducials = u.get_knPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, bin_fraction, return_pos=True, relerr=False)
		
		feature = np.hstack([fiducials, feature])
		
		if features is None:
			features = feature
			star_char = stars_to_observe
		else:
			features = np.vstack([features, feature])
			star_char = np.vstack([star_char, stars_to_observe])

	binary_stars = star_char[:,0]
	
	###############################################################################################
	### Training
	
	ml_class.train(binary_stars, features)
	# Testing the training, just to get an idea
	proba =  ml_class.predict(features)

	ann_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba, thresholds)
	
	ann_preds = ml_class.predict(features)
	ann_metr = binfind.diagnostics.get_metrics(binary_stars, ann_preds)
	auc_ann = binfind.diagnostics.auc(ann_roc_params)
	print 'AUC training ANN:', auc_ann
	print 'TPR:', ann_metr[0]
	print 'FPR:', ann_metr[1]
	print 'F1:', ann_metr[2]
	results_train["ann"].append(np.concatenate([[crit_angsep, crit_contrast], [0.0], ann_metr, [auc_ann]]))
	
	###############################################################################################
	# Validation
	
	for ith_experience in range(n_test):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_test)

		stars_to_observe, feature, fiducials = u.get_knPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, bin_fraction, return_pos=True, relerr=False)
		
		feature = np.hstack([fiducials, feature])
		
		if features is None:
			features = feature
			star_char = stars_to_observe
		else:
			features = np.vstack([features, feature])
			star_char = np.vstack([star_char, stars_to_observe])

	binary_stars = star_char[:,0]
	## Random forest
	proba_ann = ml_class.predict_proba(features)
	ann_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba_ann, thresholds)
	auc_ann = binfind.diagnostics.auc(ann_roc_params)
	
	ann_preds, _ = ml_class.predict(features)
	ann_metr = binfind.diagnostics.get_metrics(binary_stars, ann_preds)
	print 'AUC testing ANN:', auc_ann
	print 'TPR:', ann_metr[0]
	print 'FPR:', ann_metr[1]
	print 'F1:', ann_metr[2]
	
	fig = plt.figure()
	ax = plt.subplot()

	labels = ['ANN']
	
	#for line in acf_rocs:
	#	print line[:3]
	
	binfind.plots.roc(ax, [ ann_roc_params], 
		metrics=[ann_roc_params[:,3]], 
		metrics_label=r"$F_1\ \mathrm{score}$", labels=labels)
	figfname = os.path.join(outdir, "figures", "roc_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	if show: plt.show()
	plt.close()
	"""
	###############################################################################################
	## Training with PSF reconstruct
	features = None
	gnd_truth = None
	
	for ith_experience in range(n_training):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_training)

		stars_to_observe = u.get_uknPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, n_stars, bin_fraction)
		feature = euclid.get_reconstruct_fields(recovery_n_inter, recovery_n_neighbour, 
			eps=0, truth=stars_to_observe[:,0], return_proba=True, relerr=False)
		if features is None:
			features = feature
			gnd_truth = stars_to_observe[:,0]
		else:
			features = np.vstack([features, feature])
			gnd_truth = np.concatenate([gnd_truth, stars_to_observe[:,0]])

	print gnd_truth.shape 
	print features.shape
	ml_class.train(gnd_truth, features)
	# Testing the training, just to get an idea
	proba =  ml_class.predict(features)

	ann_roc_params = binfind.diagnostics.test_thresholds(gnd_truth, proba, thresholds)
	
	ann_preds = ml_class.predict(features)
	ann_metr = binfind.diagnostics.get_metrics(gnd_truth, ann_preds)
	auc_ann = binfind.diagnostics.auc(ann_roc_params)
	print 'AUC training ANN:', auc_ann
	print 'TPR:', ann_metr[0]
	print 'FPR:', ann_metr[1]
	print 'F1:', ann_metr[2]
	results_train["ann"].append(np.concatenate([[crit_angsep, crit_contrast], [0.0], ann_metr, [auc_ann]]))

	###############################################################################################
	## Validation
	idlims = []
	for ith_experience in range(n_validation):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_validation)

		stars_to_observe = u.get_uknPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, n_stars, bin_fraction)
		
		# ANN
		ann_preds, proba_ann = euclid.reconstruct_fields(ml_class, recovery_n_inter, recovery_n_neighbour, 
			eps=0, truth=stars_to_observe[:,0], return_proba=True, relerr=False)
		ann_roc_params = binfind.diagnostics.test_thresholds(stars_to_observe[:,0], proba_ann, thresholds)
	
	
		idlims.append(binfind.utils.find_nearest(ann_roc_params[:,2], thr_fpr))
	print idlims
	idlim = int(np.median(idlims))
	print idlim
	thr = ann_roc_params[idlim, 0]
	ml_class.set_threshold(thr)
	print ml_class.threshold
		
	###############################################################################################
	## Testing
	feature = None
	
	ann_res = []
	ann_rocs = None
	for ith_experience in range(n_test):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_test)

		stars_to_observe = u.get_uknPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, n_stars, bin_fraction)
		
		# ANN
		ann_preds, proba_ann = euclid.reconstruct_fields(ml_class, recovery_n_inter, recovery_n_neighbour, 
			eps=0, truth=stars_to_observe[:,0], return_proba=True, relerr=False)
		ann_roc_params = binfind.diagnostics.test_thresholds(stars_to_observe[:,0], proba_ann, thresholds)
		ann_metr = binfind.diagnostics.get_metrics(stars_to_observe[:,0], ann_preds)
		auc_ann = binfind.diagnostics.auc(ann_roc_params)
		print 'AUC testing ANN:', auc_ann
		ann_res.append(np.concatenate([[crit_angsep, crit_contrast], [0.0], ann_metr, [auc_ann]]))
		
		if ann_rocs is None:
			ann_rocs = ann_roc_params
		else:
			ann_rocs += ann_roc_params
		
		
	ann_res = np.array(ann_res)
	
	ann_rocs /= n_test
	
	if n_test > 1:
		ann_res = np.mean(ann_res, axis=0)
	
	results_test["ann"].append(ann_res)
	
	### Plotting
	
	fig = plt.figure()
	ax = plt.subplot()

	labels = ['ANN']
	binfind.plots.roc(ax, [ ann_rocs], 
		metrics=[ann_rocs[:,3]], 
		metrics_label=r"$F_1\ \mathrm{score}$", labels=labels)
	figfname = os.path.join(outdir, "figures", "roc_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	if show: plt.show()
	plt.close()
	
	for key in results_train:
		results_train[key] = np.array(results_train[key])
		results_test[key] = np.array(results_test[key])

	binfind.utils.writepickle([results_train, results_test], os.path.join(outdir, "results_{:d}_{:1.1f}.pkl".format(int(crit_angsep*1e3), crit_contrast)))
