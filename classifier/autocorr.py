import numpy as np

from .. import method

class ACF(method.Method):
	
	def __init__(self, channel_ids, channel_weights, n_data=4):
		"""
		:param channel_ids: A dict that links the channel id to its features in the data
			example: {"e1": [1,2,3,4], "e2": [4,5,6,7], "r2": [8,9,10,11]}
		:param channel_weights: The weights of the different channel in a dict:
			example  {"e1": 0.95, "e2": 0.85, "r2": 0.8}
		:param n_data: length of the data set to auto-correlate (default: 4)
		"""
		self.channel_ids = channel_ids
		self.channel_weights = channel_weights
		self.tot_weight = 0.
		# TODO I'm sure this can be optimised
		for w in self.channel_weights.values():
			self.tot_weight += w
		
		self.n_data = n_data
		self.boundaries = None
		
	def __str__(self):
		return "Auto-Correlation Analysis"
		
	def predict_proba(self, features, boundaries=None, indiv_channel=False):
		points = 0
		indiv_points = {}
		not_saved_bound = self.boundaries is None
		if not_saved_bound:
			self.boundaries = {}
			
		for ch in self.channel_ids:
			acf = self._acf(features[:,self.channel_ids[ch]])
			sacf = np.sum(acf, axis=1)
			
			if boundaries is None:
				if not_saved_bound:
					boundary = np.percentile(sacf, 95)
					self.boundaries[ch] = boundary
				else:
					boundary = self.boundaries[ch]
			else:
				boundary = boundaries[ch]
				
			sacf = self._norm_points(sacf, boundary)
			
			points += sacf * self.channel_weights[ch]
			indiv_points[ch] = sacf
		points /= (3. * self.tot_weight)
		
		if indiv_channel:
			return points, indiv_points
		
		return points
	
	def _acf(self, data):
		# TODO: can be optimised?
		result = [np.correlate(x, x, mode='full')[self.n_data-1:] for x in data]
		return np.array(result)
	
	def _norm_points(self, points, boundaries):
		points = np.clip(points, 0, boundaries)
		return points / boundaries
		
