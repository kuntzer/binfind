"""
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

from sklearn import ensemble

from .. import method

class RandomForestClassifier(method.Method):
	
	def __init__(self, *args, **params):
		self.classifier = ensemble.RandomForestClassifier(*args, **params)		
		
	def __str__(self):
		return "RandomForestClassifier from scikit-learn.org"
		
	def train(self, truth, features, *args, **params):
		self.classifier.fit(features, truth, *args, **params)
		
	def predict(self, features):
		return self.classifier.predict_proba(features)[:,1]
