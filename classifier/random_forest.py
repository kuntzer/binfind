"""
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

import numpy as np
from sklearn import ensemble

from .. import method

class RandomForestClassifier(method.Method):
	
	def __init__(self, *args, **params):
		self.classifier = ensemble.RandomForestClassifier(*args, **params)		
		
	def __str__(self):
		return "RandomForestClassifier from scikit-learn.org"
		
	def train(self, truth, features, *args, **params):
		self.classifier.fit(features, truth, *args, **params)
		
	def predict_proba(self, features):
		return self.classifier.predict_proba(features)[:,1]

	def get_feature_importance(self):
		importances = self.classifier.feature_importances_
		std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_],
		             axis=0)
		indices = np.argsort(importances)[::-1]
		return indices, importances[indices], std[indices]
