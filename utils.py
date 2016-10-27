import cPickle as pickle
import gzip
import numpy as np
import os

def writepickle(obj, filepath, protocol = -1):
	"""
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath, 'wb')
	else:
		pkl_file = open(filepath, 'wb')
	
	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	
def readpickle(filepath):
	"""
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath,'rb')
	else:
		pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	return obj

def find_nearest(array,value):
	""" Find nearest value is an array """
	idx = (np.abs(array-value)).argmin()
	return idx

def mkdir(somedir):
	"""
	A wrapper around os.makedirs.
	:param somedir: a name or path to a directory which I should make.
	"""
	if not os.path.isdir(somedir):
		os.makedirs(somedir)

def classify(pred, threshold):
	classification = np.zeros_like(pred)
	classification[pred >= threshold] = 1
	return classification

