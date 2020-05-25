#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
# import sys
# import yaml
# import math
import time
from random import Random
from utilities.utils import printStats, EV_Stats
from Population import Population
from Individual import Individual
from multiprocessing import Pool

class EV:

	def __init__(self, config, observed_sequence, obs_seq = None):
		self.config = config
		self.obs_seq = obs_seq

		#start random number generators
		uniprng = Random()
		uniprng.seed(self.config.randomSeed)
		normprng = Random()
		normprng.seed(self.config.randomSeed)

		Individual.minLimit = self.config.minLimit
		Individual.maxLimit = self.config.maxLimit
		Individual.nHiddenStates = self.config.nHiddenStates
		Individual.nObservableStates = self.config.nObservableStates
		Individual.learningRate = self.config.learningRateGlobal
		Individual.uniprng = uniprng
		Individual.normprng = normprng
		Individual.observed_sequence = observed_sequence

		Population.uniprng = uniprng
		Population.crossoverFraction = self.config.crossoverFraction

		Population.pool = Pool()

	def run(self):
		#create initial Population (random initialization)
		population = Population(self.config.populationSize, self.obs_seq)
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
		return stats


