from __future__ import division

from astropy import units as u
import numpy as np

from stars import Star 

import logging
logger = logging.getLogger(__name__)


class Catalog():
	"""
	Class containing all of the heavy lifting required to get to a catalog with a realistic-ish number of binairies
	"""
	def __init__(self, crit_angsep, crit_contrast, name='fake'):
		self.crit_angsep = crit_angsep
		self.crit_contrast = crit_contrast
		self.name = name
		
	def select_stars(self, data, m_min, m_max, increase_proba=True, verbose=False):
		
		mag_error = 0
		mass_error = 0
			
		count_stars = 0
		stars_to_use = []
		singles = []
		binaries = []
		
		lend = data.shape[0]
			
		# Prepare a fake catalogue
		for ii, star in enumerate(data):
		
			# Ditch massive stars since we don't have any data (there's not a lot of them anyway)
			if star[7]>1.5: 
				mass_error+=1
				continue
			
			# take the I magnitude
			# Beware I in Bescancon is actually i(AB) : (from Bescancon Help)
			# See http://www.astro.utoronto.ca/~patton/astro/mags.html#conversions 
			# (UBV are in the Johnson's system while RI are in the Cousins, JHKL from Koornneef.)
			IAB=star[12]-0.296 # The 12th column is already in i magnitude
			
			# Removing stars that are out of magnification cuts
			if IAB<m_min or IAB>m_max: 
				mag_error+=1
				continue
		
			# Initiate the Star class and compute the companion (if any)
			if increase_proba:
				smf = np.clip(self.crit_angsep * 200, 1., 2.) # scaling
			else:
				smf = 1 
			s = Star(dist=star[0]*1e3*u.pc, Mv=star[1], mass=star[7]*u.solMass,verbose=verbose, scaling_mult_freq=smf)
			
			has_companion=s.has_companion()
			# Not simply if has_companion as it can be also a list with the parametersFalse
			if not has_companion is False:
				dc=has_companion[1]
				# Appending infos: flux, mass of primary [SolMass], distance to system [pc], contrast, separation, id of star
				stars_to_use.append([IAB, s.mass.value, s.dist.value, has_companion[0], dc.value, ii])
				binaries.append(ii)
			else:
				stars_to_use.append([IAB, s.mass.value, s.dist.value, -1, -1, ii])
				singles.append(ii)
				
			count_stars+=1
		
		logger.info("Selected {} out of {} stars, {:1.1f}% doubles. Mass rejection: {}; Mag rejection: {}".format(
			count_stars, lend, len(binaries)/count_stars*100, mass_error, mag_error))
		
		self.stars = np.array(stars_to_use)
		self.singles = np.array(singles)
		self.binaries = np.array(binaries)
		
	def draw_catalog(self, test_angsep=None, test_contrast=None):
		
		bin_caracteristics = []
		
		if test_contrast is None:
			def test_contrast(con, crit_contrast):
				return con <= crit_contrast
		
		if test_angsep is None:
			def test_angsep(sep, crit_angsep):
				return sep >= crit_angsep
		
		# Now, for each star, get the position of the binary
		for cat_star in self.stars:
			
			# Short-hand notation
			sep = cat_star[4]
			con = cat_star[3]
			
			if test_angsep(sep, self.crit_angsep) and test_contrast(con, self.crit_contrast):
				# Convert from arcsec to small px (1/12 Euclid px)
				# TODO: Bad, it's hard-coded!!!!
				dpx = (0.1/(12))
				r = sep / dpx
		
				phi = np.random.uniform(0.,2.*np.pi)
				dx = r*np.cos(phi)
				dy = r*np.sin(phi)
			
				# This is for the LOS and inclination:
				LOSa = np.random.uniform(0,np.pi/2.)
				LOSb = np.random.uniform(0,np.pi/2.)
			
				dx *= np.cos(LOSa)
				dy *= np.cos(LOSb)
							
				r = np.hypot(dx, dy)
								
				if test_angsep(r * dpx, self.crit_angsep):
					is_bin = True
					# If we have a separation larger than 0.5 Euclid pixel, it's outside of what we computed,
					# let's shout an error (but do not raise it) and clip the rslt
					if r > 6:
						r_old = r 
						rfact = np.random.uniform(0.1, 4.)
						dx /= (np.sqrt(r/rfact))
						dy /= (np.sqrt(r/rfact))
						r = np.hypot(dx, dy)
						message = 'Got a radius larger than 6 small px: {:1.3f}\tnew r: {:1.3f}'.format(r_old, r)
						logger.warning(message)
					
					if test_angsep(r * dpx, self.crit_angsep):
						r = self.crit_angsep / dpx
					bin_caracteristics.append([1, r, con, dx, dy])
				else:
					is_bin = False
			else:
				is_bin = False
				
			if not is_bin:
				# Remembering that the star isn't a binary
				bin_caracteristics.append([0, -1, -1, 0, 0])
				
		bin_caracteristics = np.array(bin_caracteristics)
		
		nbin = bin_caracteristics[:,0].sum()
		nstars = bin_caracteristics.shape[0]
		logger.info("Prepared {} stars, {:1.1f}% doubles".format(nstars, nbin/nstars*100))
		
		return bin_caracteristics