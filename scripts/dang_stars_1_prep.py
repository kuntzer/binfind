import itertools
import numpy as np
import os

import binfind
import utils as u

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s(%(funcName)s): %(message)s', level=logging.DEBUG)

###################################################################################################
### PARAMETERS

## Simulation parameters
n_angsep = 15
n_con = 15
# Minimum separation of the stars to be qualified as binaries
crits_angsep = np.linspace(0.001, 0.015, n_angsep)#[0.001,0.003]#[0.005]
# Max contrast to be qualified as binaries
crits_contrast = np.linspace(0.1, 1.5, n_con)#np.linspace(0.1, 0.015, 7)
n_draws = 20
# Outdir
outdir = 'data/binfind_percent_meas/dang_stars_simple'

n_exposures = 4

## Observables and quality parameters
# Stellar catalogue path and parameters
star_catalogues_path = '/home/kuntzer/workspace/Blending_PSF_Euclid/data/BGM/'
l, b = (180, 45)
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
ei_max_error = 1e-8 # = 1% error
r2_max_error = 1e-8 # = 5% error
# Number of exposures for the field

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

dcon = crits_contrast[1] - crits_contrast[0]
dan = crits_angsep[1] - crits_angsep[0]
def test_angsep(sep, cangsep, dan=dan):
	condition = sep <= cangsep #and sep <= cangsep + dan
	return condition

def test_contr(con, ccontrast, dcon=dcon):
	return con >= ccontrast #and con >= ccontrast - dcon

e1_deform = []
e2_deform = []
r2_deform = []
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
	print "Running experiments on alpha < %0.4f, contrast > %0.1f --- (%d/%d)" % (crit_angsep, crit_contrast, iix+1, len(criteria))

	sim_cat = binfind.simulation.Catalog(crit_angsep, crit_contrast)
	
	features = None
	

	###################################################################################################
	### CORE OF CODE
	
	for ith_experience in range(n_draws):
		
		print '>> REALISATION %d/%d <<' % (ith_experience + 1, n_draws)

		sim_cat.select_stars(data, m_min, m_max)
		stars_to_observe = sim_cat.draw_catalog(test_angsep=test_angsep, test_contrast=test_contr)
		binary_stars = stars_to_observe[:,0]
		
		selected_stars = np.where(binary_stars == 1)[0]
		stars_to_observe = stars_to_observe[selected_stars]
	
		euclid.observe(stars_to_observe, n_exposures)
		_, feature = euclid.substract_fields()
		
		if features is None:
			features = feature
			star_char = stars_to_observe
		else:
			features = np.vstack([features, feature])
			star_char = np.vstack([star_char, stars_to_observe])

	e1_deform.append(np.median(features[:4]))
	e2_deform.append(np.median(features[4:8]))
	r2_deform.append(np.median(features[8:]))
	
e1_deform = np.array(e1_deform).reshape([n_angsep, n_con])
e2_deform = np.array(e2_deform).reshape([n_angsep, n_con])
r2_deform = np.array(r2_deform).reshape([n_angsep, n_con])

obj_to_save = [crits_angsep, crits_contrast, e1_deform, e2_deform, r2_deform]

if not os.path.exists(outdir):
	binfind.utils.mkdir(outdir)
binfind.utils.writepickle(obj_to_save, os.path.join(outdir, "dang_stars.pkl"))
