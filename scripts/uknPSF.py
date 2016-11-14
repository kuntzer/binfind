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
n_test = 20
# Number of stars per field 
n_stars = 280
# Bin fraction to reach
bin_fraction = 0.3
# Outdir
outdir = 'data/binfind_percent_meas/ukn_PSF'

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
recovery_n_inter = 5
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

		stars_to_observe, feature = u.get_knPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, bin_fraction, eps=0.001)
		
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
	proba = acf_class.predict_proba(features)
	proba_acf = proba
	roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba, thresholds)
	
	idmax = np.argmax(roc_params[:,3])
	idlim_acf = binfind.utils.find_nearest(roc_params[:,2], thr_fpr)
	
	thr = roc_params[idlim_acf, 0]
	acf_class.set_threshold(thr)
	auc_acf = binfind.diagnostics.auc(roc_params)
	print 'AUC training ACF:', auc_acf
	print 'TPR:', roc_params[idlim_acf,1]
	print 'FPR:', roc_params[idlim_acf,2]
	print 'F1:', roc_params[idlim_acf,3]
	results_train["acf"].append(np.concatenate([[crit_angsep, crit_contrast], roc_params[idlim_acf], [auc_acf]]))

		
	## Random forest
	rf_class.train(binary_stars, features)
	# Testing the training, just to get an idea
	
	proba = rf_class.predict_proba(features)
	rf_roc_params = binfind.diagnostics.test_thresholds(binary_stars, proba, thresholds)
	
	rf_preds = rf_class.predict(features)
	rf_metr = binfind.diagnostics.get_metrics(binary_stars, rf_preds)
	auc_rf = binfind.diagnostics.auc(rf_roc_params)
	print 'AUC training RF:', auc_rf
	print 'TPR:', rf_metr[0]
	print 'FPR:', rf_metr[1]
	print 'F1:', rf_metr[2]
	results_train["rf"].append(np.concatenate([[crit_angsep, crit_contrast], [0.0], rf_metr, [auc_rf]]))
	
	###############################################################################################
	
	feature = None
	
	acf_res = []
	rf_res = []
	acf_rocs = None
	rf_rocs = None
	for ith_experience in range(n_test):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_test)

		stars_to_observe = u.get_uknPSF_realisation(data, sim_cat, euclid, n_exposures, \
			m_min, m_max, n_stars, bin_fraction)
		
		# ACF
		if crit_angsep > .013:
			eps_acf = 0.036
		elif crit_angsep >= .01:
			eps_acf = 0.03
		else:
			eps_acf = 0.015
		_, proba_acf = euclid.reconstruct_fields(acf_class, recovery_n_inter, recovery_n_neighbour, eps=eps_acf, truth=stars_to_observe[:,0], return_proba=True)
		acf_roc_params = binfind.diagnostics.test_thresholds(stars_to_observe[:,0], proba_acf, thresholds)

		auc_acf = binfind.diagnostics.auc(acf_roc_params)
		print 'AUC testing ACF:', auc_acf
		acf_res.append(np.concatenate([[crit_angsep, crit_contrast], acf_roc_params[idlim_acf], [auc_acf]]))
		
		# RF
		if crit_angsep > .01:
			eps_rf = 0.07
		elif crit_angsep > .005:
			eps_rf = 0.06
		else:
			eps_rf = 0.03
		rf_preds, proba_rf = euclid.reconstruct_fields(rf_class, recovery_n_inter, recovery_n_neighbour, 
										eps=eps_rf, truth=stars_to_observe[:,0], return_proba=True)
		rf_roc_params = binfind.diagnostics.test_thresholds(stars_to_observe[:,0], proba_rf, thresholds)
		rf_metr = binfind.diagnostics.get_metrics(stars_to_observe[:,0], rf_preds)
		auc_rf = binfind.diagnostics.auc(rf_roc_params)
		print 'AUC testing RF:', auc_rf
		rf_res.append(np.concatenate([[crit_angsep, crit_contrast], [0.0], rf_metr, [auc_rf]]))
		
		if acf_rocs is None:
			acf_rocs = acf_roc_params
			rf_rocs = rf_roc_params
		else:
			acf_rocs += acf_roc_params
			rf_rocs += rf_roc_params
		
		
	acf_res = np.array(acf_res)
	rf_res = np.array(rf_res)
	
	acf_rocs /= n_test
	rf_rocs /= n_test
	
	if n_test > 1:
		acf_res = np.mean(acf_res, axis=0)
		rf_res = np.mean(rf_res, axis=0)
	
	results_test["acf"].append(acf_res)
	results_test["rf"].append(rf_res)
	
	### Plotting
	fig = plt.figure()
	ax = plt.subplot()
		
	binfind.plots.hist(ax, stars_to_observe, proba_acf >= acf_class.threshold)
	figfname = os.path.join(outdir, "figures", "complet_acf_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	
	fig = plt.figure()
	ax = plt.subplot()
	binfind.plots.hist(ax, stars_to_observe, rf_preds)
	figfname = os.path.join(outdir, "figures", "complet_rf_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	
	fig = plt.figure()
	ax = plt.subplot()

	labels = ['ACF', 'RF']
	
	#for line in acf_rocs:
	#	print line[:3]
	
	binfind.plots.roc(ax, [acf_rocs, rf_rocs], 
		metrics=[acf_rocs[:,3], rf_rocs[:,3]], 
		metrics_label=r"$F_1\ \mathrm{score}$", labels=labels)
	figfname = os.path.join(outdir, "figures", "roc_sep{:.0f}_con{:.0f}".format(crit_angsep*1e3, crit_contrast*10))
	binfind.plots.figures.savefig(figfname, fig, fancy=True, pdf_transparence=True)
	if show: plt.show()
	plt.close()
	
for key in results_train:
	results_train[key] = np.array(results_train[key])
	results_test[key] = np.array(results_test[key])

binfind.utils.writepickle([results_train, results_test], os.path.join(outdir, "results.pkl"))
