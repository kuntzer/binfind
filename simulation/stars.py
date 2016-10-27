from astropy import units as u
from astropy import constants as const
import scipy.stats 

import random
import numpy as np

def log10normal(mean, sigma, size=1):
	Z = scipy.stats.norm.rvs(size=size)
	X = mean * np.log(10) + sigma * Z / np.log(10)
	X = np.exp(X)

	return X

class Star():
	
	def __init__(self, dist, Mv, mass, scaling_mult_freq=1.,verbose=True):
		"""
		:param dist: The distance of the star in kpc
		:param Mv: The absolute magnitude
		:param mass: Mass in solar mass
		:param scaling_mult_freq: a constant scaling for the probability of the multiplicity frequency (MF in code)
		"""
		self.Mv=Mv
		self.dist=dist
		self.mass=mass
		self.scaling_mult_freq=scaling_mult_freq
		self.verbose=verbose
		self.q = None
	
	def _get_coeff(self, m):
		"""
		Returns the coefficients for the M-L relation
		http://adsabs.harvard.edu/abs/1981A%26AS...46..193H
		"""
		m=np.log10(m.value)
		if m > 0.13:
			a=-6.41 
			b=4.
		elif m>-0.62:
			a=-12.47
			b=4.8
		else:
			a=-5.61
			b=9.02
		return a,b
	
	def Mv2Mass(self, a, b, Mv):
		return 10**((Mv-b)/a)
	
	def mass2DMv(self, q):
		"""
		Computes the relation M->L
		http://adsabs.harvard.edu/abs/1981A%26AS...46..193H
		"""
		a1,b1=self._get_coeff(self.mass)
		
		dMv=(a1*np.log10(self.mass.value*q)+b1)-(a1*np.log10(self.mass.value)+b1)
		
		assert dMv > 0
		
		if self.verbose: print 'dMv', dMv

		return dMv
	
	def has_companion(self, force=False):
		"""
		Computes if the stars has a companion or not as suggested in
		http://adsabs.harvard.edu/abs/2013ARA%26A..51..269D, Tab. 1
		"""
		if self.mass is None: self.compute_mass()
		
		if self.mass <= 0.1 * u.solMass:
			MF=.22
			gamma={"std":4.2}
			a={"abar":4.5*u.au,"sigma":0.5}
		elif self.mass <= 0.5 * u.solMass:
			MF=.26
			gamma={"std":0.4}
			a={"abar":5.3*u.au,"sigma":1.3}
		# ! There is no data for 0.5 < M/Msol < 0.7
		elif self.mass <= 1.5 * u.solMass: # should end at 1.3Msun, but let's include up to 1.5
			MF=.44
			gamma={"std":0.3}
			a={"abar":45*u.au,"sigma":2.3}
		# Again no data for 1.3 < M/Msol < 1.5
		elif self.mass <= 5 * u.solMass:
			MF=.50 # This is a lower bound
			gamma={"std":-0.5}
		# No data for 5 < M/Msol < 8
		elif self.mass <= 16 * u.solMass:
			MF=.60 # lower bound too
			gamma={"std":-0.5} # No data at all, extrapolating...
		else:
			MF=.80 # lower bound too
			gamma={"shortP":-0.1,"longP":-0.5}
	
		real=random.uniform(0.,1.)
		
		if real <= MF*self.scaling_mult_freq or force:
			# What is the stellar ratio ?
			if self.mass <= 16. * u.solMass:
				q = np.random.power(gamma['std']+1)*.9+.1 # stellar masses ratio is > 0.1
				# Get the Mv of the companion
				if q < 0 or q >1:
					raise ValueError("q can only be 0<q<=1...")
				contrast = self.mass2DMv(q)
			else:
				raise NotImplemented()
			self.q = q
			
			# what is the semi-major axis ?
			if self.mass <= 1.5 * u.solMass:
				# First convert abar to period
				
				Pbar=2.*np.pi*np.sqrt((a["abar"].to(u.au))**3/(const.G*self.mass.to(u.kg)*(1.+q)))
				Pbar=Pbar.to(u.day)
				if self.verbose: print Pbar.value/365,np.log10(Pbar.value)
				
				sep=log10normal(np.log10(Pbar.value), a["sigma"])
				
				if self.verbose: 
					print '*'*3, a["abar"],'*'*3
					print sep

				sep=sep * u.day
					
				
				sep=sep.to(u.s)
				sep=((sep/2./np.pi)**2*const.G*self.mass.to(u.kg)*(1.+q))**(1/3.)
				sep=sep.to(u.au)
				if self.verbose: 
					print 'Lin separation', sep
					print a["abar"]
				# Compute the angular size
				angle=sep.to(u.pc)/self.dist.to(u.pc)*u.rad
				
				# We want quite a peaky distribution
				if np.abs(sep-a["abar"]) > 1.5*a["abar"]:
					#print 'restart !', np.abs(sep-a["abar"]), a["abar"]
					if self.verbose: 
						print "Bad value of linear separation, restarting..." 
					return self.has_companion(force=True)
			else:
				raise NotImplemented()
			

			return contrast, angle.to(u.arcsec)[0]
		else:
			return False
		
def main():
	s = Star(4.5*u.lyr, 12.3, mass=0.26*u.solMass)
	hc = False
	count = 0
	while hc == False:
		count+=1
		print count

		hc = s.has_companion()
	print 'q', s.q
	print 'contrast, sep', hc
		

if __name__=="__main__": 
	main()
