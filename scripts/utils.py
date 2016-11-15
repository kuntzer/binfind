import numpy as np

import logging
logger = logging.getLogger(__name__)

def get_knPSF_realisation(data, sim_cat, euclid, n_exposures, m_min, m_max, bin_fraction, return_pos=False, **kwargs):

	sim_cat.select_stars(data, m_min, m_max)
	stars_to_observe = sim_cat.draw_catalog()
	binary_stars = stars_to_observe[:,0]
	
	# Enforce the binary fraction now
	nb_singles = int(binary_stars.sum() / bin_fraction * (1.-bin_fraction))
	idbin = np.where(binary_stars == 1)[0]
	idsin = np.where(binary_stars == 0)[0]
	
	selected_single = np.random.choice(idsin, nb_singles, replace=False)
	selected_stars = np.concatenate([idbin, selected_single])
	
	stars_to_observe = stars_to_observe[selected_stars]

	euclid.observe(stars_to_observe, n_exposures)
	positions, feature = euclid.substract_fields(**kwargs)
	
	if return_pos:
		return stars_to_observe, feature, positions
	
	return stars_to_observe, feature

def get_uknPSF_realisation(data, sim_cat, euclid, n_exposures, m_min, m_max, n_stars, bin_fraction):
	
	sim_cat.select_stars(data, m_min, m_max)
	stars_to_observe = sim_cat.draw_catalog()
	binary_stars = stars_to_observe[:,0]
	
	# Enforce the binary fraction now
	n_bin = int(bin_fraction * n_stars)
	idbin = np.where(binary_stars == 1)[0]
	if np.size(np.where(binary_stars == 1)[0]) > n_bin:
		idbin = np.random.choice(idbin, size=n_bin, replace=False)
	idsin = np.random.choice(np.where(binary_stars == 0)[0], size=n_stars-len(idbin), replace=False)
	
	selected_stars = np.concatenate([idbin, idsin])
	
	stars_to_observe = stars_to_observe[selected_stars]
	
	euclid.observe(stars_to_observe, n_exposures)
	
	return stars_to_observe

def get_id_catalogs(crit_angsep, crit_contrast):
	if crit_contrast >= 0.5:
		if crit_angsep < 0.005: 
			fids = [1,]
		elif crit_angsep < 0.009: 
			fids = [1, 2,]
		elif crit_angsep < 0.013: 
			fids = [1, 2, 3, 4]
		else: 
			fids = [1, 2, 3, 4, 5]
	else:
		fids = [1, 2, 3, 4, 5]
		
	return fids

def get_id_catalogs_inverted(crit_angsep, crit_contrast):
	if crit_contrast <= 0.7:
		if crit_angsep < 0.005: 
			fids = [1, 2, 3, 4, 5]
		elif crit_angsep < 0.007: 
			fids = [1, 2, 3, 4]
		elif crit_angsep < 0.01: 
			fids = [1, 2, 3]
		elif crit_angsep < 0.012: 
			fids = [1, 2]
		else: 
			fids = [1]
	else:
		fids = [1, 2, 3, 4, 5]
		
	return fids
