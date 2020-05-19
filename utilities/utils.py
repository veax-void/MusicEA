import pickle
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean,stdev
from midi_utils import midi_to_csv, csv_to_notes, notes_to_midi

#Print some useful stats to screen
def printStats(pop,gen):
	print('Generation:',gen)
	avgval=0
	maxval=pop[0].fit 
	sigma=pop[0].sigma
	for ind in pop:
		avgval+=ind.fit
		if ind.fit > maxval:
			maxval=ind.fit
			sigma=ind.sigma
		#print(ind)

	print('Max fitness',maxval)
	print('Sigma',sigma)
	print('Avg fitness',avgval/len(pop))
	print('')

def generate_prob_vector(length):
	rand = np.random.rand(length)
	rand /= rand.sum(axis = 0)
	return rand

def normalize(x):
	x /= x.sum(axis = 0)
	return x

def save_model(model, filename):
	with open(filename, 'wb') as f:
		pickle.dump(model, f)

def extract_notes(filename, save = False):
	midi_filename = filename

	csv_midi = midi_to_csv(midi_filename)
	print("csv_midi.shape: ", csv_midi.shape, '\n')

	notes = csv_to_notes(csv_midi)
	print("notes.shape", notes.shape, '\n')
	
	if save:
		new_mini_name = 'ten_notes.mid'
		ten_notes = np.zeros(notes.shape, dtype='int') + 10
		ten_notes = np.array(ten_notes, dtype='str')
	
		midi_file = notes_to_midi(midi_filename, ten_notes, new_mini_name)
	
	return csv_midi, notes

#Basic class for computing, managing, plotting EV3 run stats
class EV_Stats:
	def __init__(self):
		self.bestState=[]
		self.bestFit=[]
		self.meanFit=[]
		self.stddevFit=[]
		self.mutationStrength = []

	def accumulate(self,pop):
		#find state with max fitness
		max_fit=pop[0].fit
		max_fit_state=pop[0].x
		mut_strength = pop[0].sigma
		for p in pop:
			if p.fit > max_fit: 
				 max_fit=p.fit
				 max_fit_state=p.x
				 mut_strength = p.sigma
		self.bestState.append(max_fit_state)
		self.bestFit.append(max_fit)
		self.mutationStrength.append(mut_strength)

		#compute mean and stddev
		fits=[p.fit for p in pop]
		self.meanFit.append(mean(fits))
		self.stddevFit.append(stdev(fits))

	def print(self, gen=None):
		#if gen not specified, print latest
		if gen is None: gen=len(self.bestState)-1
		print('Generation:',gen)
		print('Best fitness  : ',self.bestFit[gen])
		#print('Best state    : ',self.bestState[gen])
		print('Mean fitness  : ',self.meanFit[gen])
		print('Stddev fitness: ',self.stddevFit[gen])
		print('Mutation Strength: ',self.mutationStrength[gen])
		print('')

	def plot(self):
		#plot stats to screen & file using matplotlib
		gens=range(len(self.bestState))

		plt.figure()
		#create stacked plots (4x1)
		plt.subplots_adjust(hspace=0.5)
		plt.subplot(411)
		plt.plot(gens,self.bestFit)
		plt.ylabel('Max Fit')
		plt.title('EV Run Statistics')
		plt.subplot(412)
		plt.plot(gens,self.meanFit)
		plt.ylabel('Mean Fit')
		plt.subplot(413)
		plt.plot(gens,self.stddevFit)
		plt.ylabel('Stddev Fit')
		plt.subplot(414)
		plt.plot(gens,self.mutationStrength)
		plt.ylabel('Mutation Strength')

		#write plots to .png file, then display to screen
		plt.savefig('ev3a.png')
		plt.show()

	def __str__(self):
		s=''
		s+='bestFits  : ' + str(self.bestFit) + '\n'
		#s+='bestStates: ' + str(self.bestState) + '\n'
		s+='meanFits  : ' + str(self.meanFit) + '\n'
		s+='stddevFits: ' + str(self.stddevFit) + '\n'
		s+=':mutationStrength ' + str(self.mutationStrength)
		return s
