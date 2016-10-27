"""
A container class for all the algorithms to provide common io methods and other utilities.
"""

import os

import utils

import logging
logger = logging.getLogger(__name__)

class Method:
	
	def __init__(self):
		pass
	
	def get_name(self):
		return self.__class__.__name__
	
	def save(self, filepath=None):
		if filepath is None:
			filepath = self.workdir
		utils.mkdir(filepath)
		filename = os.path.join(filepath, "%s.pkl" % self.get_name())
		utils.writepickle(self, filename)
		logger.info("Saved method %s to %s" % (self.get_name(), filename))
		
	def set_workdir(self, workdir):
		"""
		sets the workdir
		"""
		self.workdir = workdir
		utils.mkdir(self.workdir)
	
	def info(self):
		"""
		Prints out all the variables
		"""
		import inspect
	
		message = "All variables available for method %s" % self.__str__()
		print message
		print '-'*len(message)
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		for a in attributes:
			if (a[0].startswith('__') and a[0].endswith('__')): continue
			print a[0], "=", a[1]

	def _set_default(self, dico, name, default, warn=None):
		if not name in dico.keys():
			dico[name] = default
			
		if not warn is None:
			logger.warn(warn)

