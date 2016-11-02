import numpy as np
import sklearn.metrics as metrics

import utils

def test_thresholds(truth, proba, thresholds):
	"""
	Takes the probabilities as inputs and different thresholds to evaluate the performance.
	
	:param truth: The ground truth in a 1D array
	:param proba: The predictions to be tested
	
	:returns: A table with the following metrics: [thr, tpr, fpr, f1, recall, precision, accuracy]
	"""
	results = []
	for thr in thresholds:
		predictions = utils.classify(proba, thr)
		
		tpr, fpr, f1, recall, precision, accuracy = get_metrics(truth, predictions)
		results.append([thr, tpr, fpr, f1, recall, precision, accuracy])
	
	return np.array(results)

def get_metrics(truth, predictions):
	tpr, fpr = get_tpr_fpr(truth, predictions)
	f1 = metrics.f1_score(truth, predictions, average='binary')
	recall = metrics.recall_score(truth, predictions, average='binary')
	precision = metrics.precision_score(truth, predictions, average='binary')
	accuracy = metrics.accuracy_score(truth, predictions)
	
	return [tpr, fpr, f1, recall, precision, accuracy]

def get_tpr_fpr(truth, pred):
	"""
	Returns the Sensitivity (True Positive Rate) and the Specificity (False Positive Rate) for two
	numpy arrays: a `pred` and a `truth`.
	"""
	
	if not pred.shape == truth.shape :
		raise IndexError("The prediction and the truth arrays must have the same dimensions")
	
	n_true = np.where(truth == 1)[0].size
	n_false = np.where(truth == 0)[0].size
	
	# Sensitivity
	n_true_pos = np.where(np.logical_and(truth == 1, pred == 1))[0].size
	tpr = float(n_true_pos) / float(n_true)
	
	# Specificity
	n_false_pos = np.where(np.logical_and(truth == 0, pred == 1))[0].size
	fpr = float(n_false_pos) / float(n_false)
	
	return tpr, fpr

def get_unique_tpr_fpr(params, id_fpr=2, id_tpr=1, return_indx=False):
	xx = np.unique(params[:,id_fpr], return_index=True)[1]
	tpr_ = np.concatenate([[0], params[xx, id_tpr], [1]])
	fpr_ = np.concatenate([[0], params[xx, id_fpr], [1]])
	
	if return_indx:
		return tpr_, fpr_, xx
		
	return tpr_, fpr_

def auc(params, id_fpr=2, id_tpr=1):
	tpr_, fpr_ = get_unique_tpr_fpr(params, id_fpr, id_tpr)
	return np.trapz(tpr_, x=fpr_)
	