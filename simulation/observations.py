from scipy.spatial import cKDTree
import numpy as np

from .. import utils

import logging
logger = logging.getLogger(__name__)


class Observations():
	
	def __init__(self, ei_max_error, r2_max_error, fname_interpolation, fname_fiducial, radius=6):
		self.ei_max_error = ei_max_error
		self.r2_max_error = r2_max_error
		
		psf_positions = np.loadtxt(fname_fiducial)
		x_psf = psf_positions[:,0]
		y_psf = psf_positions[:,1]
		self.min_x_psf = np.amin(x_psf)
		self.min_y_psf = np.amin(y_psf)
		self.max_x_psf = np.amax(x_psf)
		self.max_y_psf = np.amax(y_psf)
		
		self.configurations, self.fields_e1, self.fields_e2, self.fields_sigma = utils.readpickle(fname_interpolation)
		# Preparing for the matching of the indexes
		self.contrasts = np.unique(self.configurations[:,0])

		dxdy = utils.rdisk(radius=radius)
		self.dxdytree = cKDTree(dxdy)
		
				# Preparing the selection of the interpolation for no binaries
		id_null = np.where(np.all(self.configurations == 0, axis=1))[0]
		# Just make sure that the size of the array is one
		assert np.size(id_null) == 1
		self.id_null = id_null[0]
		
		self.meanr2 = 0
		for x, y in zip(x_psf, y_psf):
			self.meanr2 += self.fields_sigma[self.id_null](x, y)
		self.meanr2 /= len(x_psf)
	
	def observe(self, catalog, n_exposures, delta_inbetween_frame):
		
		self.n_exposures = n_exposures
		observed_stars = []	
		# Now, for each star, get the position of the binary
		for this_star in catalog:
			
			con = this_star[2]
			dx = this_star[3]
			dy = this_star[4]
			
			# Assign a position in the field of view
			x_star = np.random.uniform(low=self.min_x_psf, high=self.max_x_psf)
			y_star = np.random.uniform(low=self.min_y_psf, high=self.max_y_psf)
			
			# Making n_exposures observations of the same stars
			obs_ss = []
			for _ in range(n_exposures):	
				if this_star[0] == 1:
					# Preparing the selection of the interpolation for no binaries
					if con > self.contrasts[-1]: 
						idcons = [utils.find_nearest(self.contrasts, con)]
						wcon = [1.]
					else:
						ds = np.abs(self.contrasts - con)
						idcons = np.argsort(ds)[:2]
						wcon = 1. / ds[idcons]
					
					e1_star = 0.
					e2_star = 0.
					sigma_star = 0.
					for ii, idcon in enumerate(idcons):
						idcon = np.where(self.configurations[:,0] == self.contrasts[idcon])[0]
						dist, ids = self.dxdytree.query([dx, dy], k=4)
						
						we = 1./dist

						e1_star += np.average([fe1(x_star, y_star) for fe1 in self.fields_e1[idcon][ids]], weights=we) * wcon[ii]
						e2_star += np.average([fe2(x_star, y_star) for fe2 in self.fields_e2[idcon][ids]], weights=we) * wcon[ii]
						sigma_star += np.average([sig(x_star, y_star) for sig in self.fields_sigma[idcon][ids]], weights=we) * wcon[ii]
					
					e1_star /= np.sum(wcon)
					e2_star /= np.sum(wcon)
					sigma_star /= np.sum(wcon)
				else:
					# Interpolate the ellipticity and size
					e1_star = self.fields_e1[self.id_null](x_star, y_star)
					e2_star = self.fields_e2[self.id_null](x_star, y_star)
					sigma_star = self.fields_sigma[self.id_null](x_star, y_star)
					
					
				# Adding some noise in the measure of e1, e2
				# prob(N>maxval) ~ 1e-5
				e1_star += np.random.normal(scale=self.ei_max_error/3.8)
				e2_star += np.random.normal(scale=self.ei_max_error/3.8)
				sigma_star += np.random.normal(scale=self.r2_max_error/3.8) * self.meanr2
					
				# Adding to the catalogue
				obs_ss.append([x_star, y_star, e1_star, e2_star, sigma_star])
				# And finally, dithering
				x_star += (float(delta_inbetween_frame[0]) * 0.1 / 360.)
				y_star += (float(delta_inbetween_frame[1]) * 0.1 / 360.)
			observed_stars.append(obs_ss)	
		self.observed_stars = np.asarray(observed_stars)
		
	def substract_fields(self):
		obs_x = self.observed_stars[:,:,0].flatten()
		obs_y = self.observed_stars[:,:,1].flatten()
		
		n_stars_obs = self.observed_stars.shape[0]
		fiducial_e1 = self.fields_e1[self.id_null](obs_x, obs_y).reshape([n_stars_obs, self.n_exposures])
		fiducial_e2 = self.fields_e2[self.id_null](obs_x, obs_y).reshape([n_stars_obs, self.n_exposures])
		fiducial_sigma = self.fields_sigma[self.id_null](obs_x, obs_y).reshape([n_stars_obs, self.n_exposures])
		
		pos = [obs_x, obs_y]
		dev_e1 = (self.observed_stars[:,:,2] - fiducial_e1) / fiducial_e1
		dev_e2 = (self.observed_stars[:,:,3] - fiducial_e2) / fiducial_e2
		dev_r2 = (self.observed_stars[:,:,4] - fiducial_sigma) / fiducial_sigma	
		
		features = np.array([dev_e1.T, dev_e2.T, dev_r2.T]).reshape([3*self.n_exposures, n_stars_obs]).T
		
		return pos, features
		
