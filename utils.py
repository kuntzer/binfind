import cPickle as pickle
import csv
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

def load_bmg(fname, main_sequence):	
	data=[]
	with open(fname+'.dat') as observability_file:
		observability_data = csv.reader(observability_file, delimiter="\t")
		for row in observability_data:
		# if line is empty, skip otherwise filter out the blank
			dline=row[0].split()
			if len(dline)==17 and not dline[6].isdigit():
				dline.insert(6, '0')
			if dline[0][0]=='#': continue
			data.append(np.asarray(dline, dtype=np.float))
			
	data=np.asarray(data)
	if main_sequence: 
		data=data[data[:,2] == 5] #Takes only main sequence stars
	
	return data

def rdisk(radius, norien=25, nrad=35):
	orientations = np.linspace(0., np.pi * 2., norien, endpoint=False)
	dtheta = (orientations[:2] / 2.)[-1]

	nrad = float(nrad)
	radii = ( np.arange(nrad) / (nrad - 1) )**2 * float(radius)

	coord = []
	seen_nought = False

	for ir, r in enumerate(radii):
		if r == 0 :
			if not seen_nought:
				coord.append([0., 0.])
				seen_nought = True
			continue

		for orientation in orientations:
			x = np.cos(orientation + dtheta * (ir % 2)) * r
			y = np.sin(orientation + dtheta * (ir % 2)) * r
			coord.append([x, y])

	return np.asarray(coord)
