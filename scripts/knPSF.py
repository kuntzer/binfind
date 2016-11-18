import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

import binfind
import utils as u

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

ch_id = {"e1": [1,2,3,4], "e2": [4,5,6,7], "r2": [8,9,10,11]}
ch_w = {"e1": 0.7, "e2": 0.55, "r2": 0.0}
n_exposures = 4
acf_class = binfind.classifier.ACF(ch_id, ch_w, n_exposures)
rf_class = binfind.classifier.RandomForestClassifier(50)
binfind.plots.figures.set_fancy()

results_train = {'acf':[], 'rf':[]}
results_test = {'acf':[], 'rf':[]}

###################################################################################################
### PARAMETERS

## Simulation parameters
# Minimum separation of the stars to be qualified as binaries
crits_angsep = np.linspace(0.001, 0.015, 15)
# Max contrast to be qualified as binaries
crits_contrast = np.linspace(0.1, 1.5, 15)
# Number of times to do the whfname_interpolationole analysis 
n_training = 15
n_test = 15
# Bin fraction to reach
bin_fraction = 0.3
# Outdir
outdir = 'data/binfind_percent_meas/kn_PSF'

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
# Number of exposures for the field

# What is the wanted False Positive Rate ? (in fraction)
thr_fpr = 0.1

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

	fids = u.get_id_catalogs(crit_angsep, crit_contrast)
			
	if len(fids) != previous_sep:	
		previous_sep = len(fids)
		data = None
		for fid in fids:
			fname= os.path.join(star_catalogues_path, 'star_field_BGM_i_%d_%d_%d' % (l, b, fid))
			datal=binfind.utils.load_bmg(fname, main_sequence=True)
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
	
	for ith_experience in range(n_training):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_training)

		stars_to_observe, feature = u.get_knPSF_realisation(data, sim_cat, euclid, n_exposures, m_min, m_max, bin_fraction)
		
		if features is None:
			features = feature
			star_char = stars_to_observe
		else:
			features = np.vstack([features, feature])
			star_char = np.vstack([star_char, stars_to_observe])

	binary_stars = star_char[:,0]
	
	###############################################################################################
	### Training

	## ACF
	proba_acf = acf_class.predict_proba(features)
	acf_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba_acf, thresholds)
	
	idlim_acf = binfind.utils.find_nearest(acf_roc_params[:,2], thr_fpr)
	
	thr = acf_roc_params[idlim_acf, 0]
	acf_class.set_threshold(thr)

	auc_acf = binfind.diagnostics.auc(acf_roc_params)
	print 'AUC training ACF:', auc_acf
	print 'TPR:', acf_roc_params[idlim_acf,1]
	print 'FPR:', acf_roc_params[idlim_acf,2]
	print 'F1:', acf_roc_params[idlim_acf,3]
	results_train["acf"].append(np.concatenate([[crit_angsep, crit_contrast], acf_roc_params[idlim_acf], [auc_acf]]))
		
	## Random forest
	rf_class.train(binary_stars, features)
	# Testing the training, just to get an idea
	proba = rf_class.predict_proba(features)
	rf_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba, thresholds)
	
	idlim_rf = binfind.utils.find_nearest(rf_roc_params[:,2], thr_fpr)
	thr = rf_roc_params[idlim_rf, 0]
	rf_class.set_threshold(thr)
	
	rf_feat_imp = rf_class.get_feature_importance()
	rf_preds = rf_class.predict(features)
	rf_metr = binfind.diagnostics.get_metrics(binary_stars, rf_preds)

	auc_rf = binfind.diagnostics.auc(rf_roc_params)
	print 'AUC training RF:', auc_rf
	print 'TPR:', rf_metr[0]
	print 'FPR:', rf_metr[1]
	print 'F1:', rf_metr[2]
	results_train["rf"].append(np.concatenate([[crit_angsep, crit_contrast], [0.], rf_metr, [auc_rf]]))
	
	### Plotting
		
	fig = plt.figure()
	ax = plt.subplot()
	binfind.plots.errorbar(ax, rf_feat_imp[1], rf_feat_imp[2], rf_feat_imp[0])
	figfname = os.path.join(outdir, "figures", "featimp_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	
	feature = None
	for ith_experience in range(n_test):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_test)

		stars_to_observe, feature = u.get_knPSF_realisation(data, sim_cat, euclid, n_exposures, m_min, m_max, bin_fraction)
		
		if features is None:
			features = feature
			star_char = stars_to_observe
		else:
			features = np.vstack([features, feature])
			star_char = np.vstack([star_char, stars_to_observe])

	binary_stars = star_char[:,0]
	
	###############################################################################################
	### Testing

	## ACF
	proba_acf, indiv = acf_class.predict_proba(features, indiv_channel=True)
	acf_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba_acf, thresholds)
		
	e1_roc_params = binfind.diagnostics.test_thresholds(binary_stars, indiv['e1'], thresholds)
	e2_roc_params = binfind.diagnostics.test_thresholds(binary_stars, indiv['e2'], thresholds)
	r2_roc_params = binfind.diagnostics.test_thresholds(binary_stars, indiv['r2'], thresholds)
	auc_acf = binfind.diagnostics.auc(acf_roc_params)
	print 'AUC testing ACF:', auc_acf
	print 'TPR:', acf_roc_params[idlim_acf,1]
	print 'FPR:', acf_roc_params[idlim_acf,2]
	print 'F1:', acf_roc_params[idlim_acf,3]
	results_test["acf"].append(np.concatenate([[crit_angsep, crit_contrast], acf_roc_params[idlim_acf], [auc_acf]]))
	
	## Random forest
	proba_rf = rf_class.predict_proba(features)
	rf_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba_rf, thresholds)
	auc_rf = binfind.diagnostics.auc(rf_roc_params)
	
	rf_preds = rf_class.predict(features)
	rf_metr = binfind.diagnostics.get_metrics(binary_stars, rf_preds)
	print 'AUC testing RF:', auc_rf
	print 'TPR:', rf_metr[0]
	print 'FPR:', rf_metr[1]
	print 'F1:', rf_metr[2]
	
	results_test["rf"].append(np.concatenate([[crit_angsep, crit_contrast], [0.0], rf_metr, [auc_rf]]))

	### Plotting
	fig = plt.figure()
	ax = plt.subplot()
	
	binfind.plots.hist(ax, star_char, proba_acf >= acf_class.threshold)
	figfname = os.path.join(outdir, "figures", "complet_acf_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	
	fig = plt.figure()
	ax = plt.subplot()
	binfind.plots.hist(ax, star_char, rf_preds)
	figfname = os.path.join(outdir, "figures", "complet_rf_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	
	fig = plt.figure()
	ax = plt.subplot()
		
	labels = ['ACF', 'N(e_1)', 'N(e_2)', 'N(R^2)', 'RF']
	binfind.plots.roc(ax, [acf_roc_params, e1_roc_params, e2_roc_params, r2_roc_params, rf_roc_params], 
		metrics=[acf_roc_params[:,3], e1_roc_params[:,3], e2_roc_params[:,3], r2_roc_params[:,3], rf_roc_params[:,3]], 
		metrics_label=r"$F_1\ \mathrm{score}$", labels=labels)
	figfname = os.path.join(outdir, "figures", "roc_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	if show: plt.show()

for key in results_train:
	results_train[key] = np.array(results_train[key])
	results_test[key] = np.array(results_test[key])

binfind.utils.writepickle([results_train, results_test], os.path.join(outdir, "results.pkl"))
