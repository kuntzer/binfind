from __future__ import division

import numpy as np
import scipy.interpolate as interp
from scipy.spatial import cKDTree
import sklearn.metrics as metrics

from .. import utils
from .. import diagnostics

import logging
logger = logging.getLogger(__name__)


class Observations():
	
	def __init__(self, ei_max_error, r2_max_error, fname_interpolation, fname_fiducial, radius=6):
		self.ei_max_error = ei_max_error
		self.r2_max_error = r2_max_error
		
		psf_positions = np.loadtxt(fname_fiducial)
		self.x_psf = psf_positions[:,0]
		self.y_psf = psf_positions[:,1]
		self.min_x_psf = np.amin(self.x_psf)
		self.min_y_psf = np.amin(self.y_psf)
		self.max_x_psf = np.amax(self.x_psf)
		self.max_y_psf = np.amax(self.y_psf)
		
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
		
		#self.meane = []
		self.meanr2 = 0
		for x, y in zip(self.x_psf, self.y_psf):
			self.meanr2 += self.fields_sigma[self.id_null](x, y)
			#e1_ = self.fields_e1[self.id_null](x, y)
			#e2_ = self.fields_e2[self.id_null](x, y)
			#self.meane.append(np.hypot(e1_, e2_))
		self.meanr2 /= len(self.x_psf)
		#print np.amin(self.meane), np.amax(self.meane)
		self.meane = 0.1
	
	def observe(self, catalog, n_exposures, delta_inbetween_frame):
		
		self.n_exposures = n_exposures
		observed_stars = []	
		count_doubles = 0
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
					count_doubles += 1./n_exposures
					
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
						dist, ids = self.dxdytree.query([dx, dy], k=3)
						we = 1./dist
						e1_star += np.average([fe1(x_star, y_star) for fe1 in self.fields_e1[idcon][ids]], weights=we) * wcon[ii]
						e2_star += np.average([fe2(x_star, y_star) for fe2 in self.fields_e2[idcon][ids]], weights=we) * wcon[ii]
						sigma_star += np.average([sig(x_star, y_star) for sig in self.fields_sigma[idcon][ids]], weights=we) * wcon[ii]
						#print e1_star; exit()
					e1_star /= np.sum(wcon)
					e2_star /= np.sum(wcon)
					sigma_star /= np.sum(wcon)
				else:
					# Interpolate the ellipticity and size
					e1_star = self.fields_e1[self.id_null](x_star, y_star)
					e2_star = self.fields_e2[self.id_null](x_star, y_star)
					sigma_star = self.fields_sigma[self.id_null](x_star, y_star)
					
				# Adding some noise in the measure of e1, e2
				#if this_star[0] == 1 :print self.fields_e2[self.id_null](x_star, y_star), e2_star, 
				"""if this_star[0] == 1 :
					#print dx, dy, e1_star, e2_star, np.hypot(e1_star, e2_star), sigma_star * 12. * 4.
					t = self.fields_e2[self.id_null](x_star, y_star)
					te = np.hypot(self.fields_e2[self.id_null](x_star, y_star), self.fields_e1[self.id_null](x_star, y_star))
					o = e2_star
					oe = np.hypot(e2_star, e1_star)
					obs_errors.append(oe-te)
					print te, oe, (oe-te)/te
					#print "%1.2f \t %1.4f %+1.1e\t%1.4f %1.4f %+1.1e" % (this_star[1] / .12,t,(o-t)/t, te,oe,(oe-te)/te),
				"""
				e1_star += np.random.normal(scale=self.ei_max_error * self.meane)
				e2_star += np.random.normal(scale=self.ei_max_error * self.meane)
				"""if this_star[0] == 1 :
					oe = np.hypot(e2_star, e1_star)
					#print "\t%1.4f %+1.1e" % (oe,(oe-te)/te)
				#if this_star[0] == 1:print e2_star"""
				sigma_star += np.random.normal(scale=self.r2_max_error * self.meanr2)
					
				# Adding to the catalogue
				obs_ss.append([x_star, y_star, e1_star, e2_star, sigma_star])
				# And finally, dithering
				x_star += (float(delta_inbetween_frame[0]) * 0.1 / 360.)
				y_star += (float(delta_inbetween_frame[1]) * 0.1 / 360.)
			observed_stars.append(obs_ss)	
		logger.info("Observed {} stars, {:1.1f}% doubles".format(len(observed_stars), count_doubles/len(observed_stars)*100))
		self.observed_stars = np.asarray(observed_stars)

	def substract_fields(self, eps=0., error_e=2e-4, error_r2=1e-3):
		obs_x = self.observed_stars[:,:,0].flatten()
		obs_y = self.observed_stars[:,:,1].flatten()
		
		n_stars_obs = self.observed_stars.shape[0]
		fiducial_e1 = self.fields_e1[self.id_null](obs_x, obs_y).reshape([n_stars_obs, self.n_exposures])
		fiducial_e2 = self.fields_e2[self.id_null](obs_x, obs_y).reshape([n_stars_obs, self.n_exposures])
		fiducial_sigma = self.fields_sigma[self.id_null](obs_x, obs_y).reshape([n_stars_obs, self.n_exposures])
		
		fiducial_e1 += np.random.normal(scale=error_e * self.meane, size=[n_stars_obs, self.n_exposures])
		fiducial_e2 += np.random.normal(scale=error_e * self.meane, size=[n_stars_obs, self.n_exposures])
		fiducial_sigma += np.random.normal(scale=error_r2 * self.meane, size=[n_stars_obs, self.n_exposures])
		
		pos = [obs_x, obs_y]
		dev_e1 = (self.observed_stars[:,:,2] - fiducial_e1) / (fiducial_e1 + eps)
		dev_e2 = (self.observed_stars[:,:,3] - fiducial_e2) / (fiducial_e2 + eps)
		dev_r2 = (self.observed_stars[:,:,4] - fiducial_sigma) / (fiducial_sigma + eps)	
		
		features = np.array([dev_e1.T, dev_e2.T, dev_r2.T]).reshape([3*self.n_exposures, n_stars_obs]).T
		
		return pos, features
	
	def reconstruct_fields(self, classifier, n_iter_reconstr, n_neighbours, eps, truth=None, return_proba=False):
		n_stars = self.observed_stars.shape[0]
		ids_all = range(n_stars)
		outliers_ids = None
		
		observed_stars = self.observed_stars
		
		for kk in range(n_iter_reconstr):
			logger.info("PSF reconstruction with {:s}, iteration {:d}/{:d}".format(classifier, kk+1, n_iter_reconstr))
			if 	np.size(outliers_ids) >= n_stars - n_neighbours:
				continue
			
			de1 = []
			de2 = []
			dsigma = []
			
			for ii in range(n_stars):
				if outliers_ids is None:
					ids_singles = ids_all
					ids_single = np.delete(ids_singles, [ii])
					
				else:
					# Remove outliers from the list
					ids_single = np.delete(ids_all, np.concatenate([outliers_ids, [ii]]))
	
				obs_x = (observed_stars[ids_single,0,0].flatten())
				obs_y = (observed_stars[ids_single,0,1].flatten())
				
				xy = np.array([obs_x, obs_y]).T
				
				ie1 = []
				ie2 = []
				isigma = []
				
				for iobs in range(self.n_exposures):
					ie1.append(interp.NearestNDInterpolator(xy, observed_stars[ids_single,iobs,2]) )
					ie2.append(interp.NearestNDInterpolator(xy, observed_stars[ids_single,iobs,3]) )
					isigma.append(interp.NearestNDInterpolator(xy, observed_stars[ids_single,iobs,4]) )
				
				tree = cKDTree(zip(obs_x, obs_y))
				d, inds = tree.query(zip([observed_stars[ii,0,0]], [observed_stars[ii,0,1]]), k = n_neighbours)
				inds = inds[d > 0]
				d = d[d > 0]
				weights = 1. / (d*2)
		
				obs_e1 = np.median(observed_stars[inds,:,2], axis=1)
				obs_e2 = np.median(observed_stars[inds,:,3], axis=1)
				obs_r2 = np.median(observed_stars[inds,:,4], axis=1)
				
				try:	
					dinterp_e1 = np.average(obs_e1, weights=weights) 
				except :
					print xy.shape
					print weights	
					print d
					print inds
					raise
									
				dinterp_e2 = np.average(obs_e2, weights=weights)
				dinterp_r2 = np.average(obs_r2, weights=weights)
				
	
				ae1 = []
				ae2 = []
				asigma = []
				for iobs in range(self.n_exposures):
					#print observed_stars[ii,iobs,2] - ie1[iobs](observed_stars[ii,0,0], observed_stars[ii,0,1]),
					#print ie1[iobs](observed_stars[ii,0,0], observed_stars[ii,0,1])
					ae1.append(ie1[iobs](observed_stars[ii,0,0], observed_stars[ii,0,1]))
					ae2.append(ie2[iobs](observed_stars[ii,0,0], observed_stars[ii,0,1]))
					asigma.append(isigma[iobs](observed_stars[ii,0,0], observed_stars[ii,0,1]))
				
				dinterp_e1 = np.median(np.asarray(ae1))
				dinterp_e2 = np.median(np.asarray(ae2))
				dinterp_r2 = np.median(np.asarray(asigma))
				
				de1.append((observed_stars[ii,:,2] - dinterp_e1) / (dinterp_e1 + eps))
				de2.append((observed_stars[ii,:,3] - dinterp_e2) / (dinterp_e2 + eps))
				dsigma.append((observed_stars[ii,:,4] - dinterp_r2) / (dinterp_r2 + eps))
		
			de1 = np.array(de1)
			de2 = np.array(de2)
			dsigma = np.array(dsigma)
			
			features = np.concatenate([de1, de2, dsigma], axis=1)

			preds = classifier.predict(features)
			outliers_ids = np.where(preds == 1)[0]
			
			if truth is not None :
				f1_ = metrics.f1_score(truth, preds, average='binary')
				tpr, fpr = diagnostics.get_tpr_fpr(truth, preds)
				msg = "F1={:1.3f}, FPR={:2.1f}%, TPR={:2.1f}%".format(f1_, fpr*100., tpr*100.)
				logger.info(msg)
		
		proba = classifier.predict_proba(features)	
		if return_proba:
			return preds, proba
		else:
			return preds
