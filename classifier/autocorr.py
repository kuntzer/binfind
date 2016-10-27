import numpy as np

from .. import method

class ACF(method.Method):
	
	def __init__(self, channel_ids, channel_weights):
		"""
		:param channel_ids: A dict that links the channel id to its features in the data
			example: {"e1": [1,2,3,4], "e2": [4,5,6,7], "r2": [8,9,10,11]}
		:param channel_weights: The weights of the different channel in a dict:
			example  {"e1": 0.95, "e2": 0.85, "r2": 0.8}
		"""
		self.channel_ids = channel_ids
		self.channel_weights = channel_weights
		self.tot_weight = 0.
		# I'm sure this can be optimised
		for w in self.channel_weights.values():
			self.tot_weight += w
		
	def __str__(self):
		return "Auto-Correlation Analysis"
		
	def predict(self, features):
		points = 0
		for ch in self.channel_ids:
			acf = self._acf(features[self.channel_ids[ch]])
			points += np.sum(acf, axis=1) * self.channel_weights[ch]
		points /= self.tot_weight
		return points

	def _acf(self, x):
		result = np.correlate(x, x, mode='full')
		return result[result.size/2:]
