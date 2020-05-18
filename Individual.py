# Individual.py
import utils
import math
import numpy as np
#from HMM import HMM
from HMM import HiddenMarkovModel as HMM

#A Individual class for HMM model
class Individual:
	"""
	Individual
	"""
	minSigma=1e-100
	maxSigma=1
	learningRate=1
	N = 2		# Number of hidden states
	M = 3		# Number of observables states
	minLimit=None
	maxLimit=None
	uniprng=None
	normprng=None
	fitFunc=None

	def __init__(self, states = None, observables = None):
		if states is None:
			self.states = ['h{}'.format(i+1) for i in range(self.__class__.N)]
		else:
			self.states = states
		if observables is None:
			self.observables = ['{}'.format(i) for i in range(self.__class__.M)]
		else:
			self.observables = observables
		self.model = HMM(self.states, self.observables)
		self.x = [self.model.get_pi(), self.model.get_A(), self.model.get_B()]
		#self.fit = self.__class__.fitFunc(self.x)
		self.fit = None
		self.sigma = []
		for _ in range(3):
			self.sigma.append(self.uniprng.uniform(0.9,0.1)) #use "normalized" sigma
		
	def crossover(self, other):
		#randomly choose position for model.pi
		pos0 = self.uniprng.choice(range(len(self.x[0])))
		#randomly choose position for model.A
		pos1 = self.uniprng.choice(range(self.x[1].shape[0]))
		#randomly choose position for model.B
		pos2 = self.uniprng.choice(range(self.x[2].shape[0]))

		# crossover for model.pi
		tmp = np.copy(other.x[0][pos0:])
		other.x[0][pos0:] = self.x[0][pos0:]
		self.x[0][pos0:] = tmp
		self.x[0] = utils.normalize(self.x[0])
		other.x[0] = utils.normalize(other.x[0])
		self.model.set_pi(self.x[0])
		other.model.set_pi(other.x[0])

		# Crossover for model.A
		tmp = np.copy(other.x[1][pos1:])
		other.x[1][pos1:] = self.x[1][pos1:]
		self.x[1][pos1:] = tmp
		self.model.set_A(self.x[1])
		other.model.set_A(other.x[1])
		
		# Crossover for model.A
		tmp = np.copy(other.x[2][pos2:])
		other.x[2][pos2:] = self.x[2][pos2:]
		self.x[2][pos2:] = tmp
		self.model.set_B(self.x[2])
		other.model.set_B(other.x[2])
		
		self.fit=None
		other.fit=None
	
	def mutate(self):
		for i in range(len(self.sigma)):
			self.sigma[i] = self.sigma[i] * math.exp(self.learningRate * self.normprng.normalvariate(0,1))
			if self.sigma[i] < self.minSigma: self.sigma[i] = self.minSigma
			if self.sigma[i] > self.maxSigma: self.sigma[i] = self.maxSigma

		# Mutation for model.pi
		if self.uniprng.random() < self.sigma[0]:
			self.x[0] = utils.generate_prob_vector(len(self.x[0]))
			self.model.set_pi(self.x[0])

		# Mutation for model.A
		for i in range(self.x[1].shape[0]):
			if self.uniprng.random() < self.sigma[1]:
				self.x[1][i] = utils.generate_prob_vector(len(self.x[1][i]))
		self.model.set_A(self.x[1])

		# Mutation for model.B
		for i in range(self.x[2].shape[0]):
			if self.uniprng.random() < self.sigma[2]:
				self.x[2][i] = utils.generate_prob_vector(len(self.x[2][i]))
		self.model.set_B(self.x[2])

		self.fit = None

	
	def evaluateFitness(self, obs_seq):
		if self.fit == None:
			self.fit = self.model.score(obs_seq)
		
	def __str__(self):
		pass
