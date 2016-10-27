import numpy as np

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
