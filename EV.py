#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
import time
import pickle
import numpy as np
import utilities.midi_utils as midi
from random import Random
from Population import Population
from Individual import Individual
from multiprocessing import Pool
from utilities.utils import printStats, EV_Stats

class EV:
	def __init__(self, config, midi_filename):
		# Get sequence
		csv_file = midi.midi_to_csv(midi_filename)
		notes = midi.csv_to_notes(csv_file)
		observed_sequence = list(notes)

		# States used in the individuals
		if not config.useFullSequence:
			observed_states = list(np.unique(notes))
		else:
			observed_states = np.arange(config.nObservableStates)

		#start random number generators
		uniprng = Random()
		uniprng.seed(config.randomSeed)
		normprng = Random()
		normprng.seed(config.randomSeed)

		# Individual init
		Individual.observed_sequence = observed_sequence
		Individual.nHiddenStates = config.nHiddenStates
		Individual.learningRate = config.learningRate
		Individual.observed_states = observed_states
		Individual.normprng = normprng
		Individual.uniprng = uniprng

		# Population init
		Population.uniprng = uniprng
		Population.crossoverFraction = config.crossoverFraction
		Population.pool = Pool()

		# Save sequence and config
		self.config = config
		self.observed_sequence = observed_sequence

	def run(self):
		#create initial Population (random initialization)
		population = Population(self.config.populationSize)
		population.evaluateFitness()

# 		print initial pop stats
		#printStats(population, 0)

		#accumulate & print stats
		stats = EV_Stats()
		stats.accumulate(population)
		stats.print()

		#evolution main loop
		for i in range(self.config.generationCount):
			gen_start = time.time()
			#create initial offspring population by copying parent pop
			offspring=population.copy()

			#select mating pool
			offspring.conductTournament()

			#perform crossover
			offspring.crossover()

			#random mutation
			offspring.mutate()

			#update fitness values
			offspring.evaluateFitness()

			#survivor selection: elitist truncation using parents+offspring
			population.combinePops(offspring)
			population.truncateSelect(self.config.populationSize)

			#print population stats
			#printStats(population,i+1)

# 			accumulate & print stats
			stats.accumulate(population)
			stats.print()
			print("[INFO] Generation {} finished in {} minutes".format(i+1, (time.time() - gen_start)/60))
# 		plot accumulated stats to file/screen using matplotlib
		stats.plot()

		with open("stat_gen{}_hid{}_state{}.pickle".format(
			self.config.generationCount, self.config.nHiddenStates, len(self.observed_sequence)), 'wb') as f:
			pickle.dump(stats, f)


