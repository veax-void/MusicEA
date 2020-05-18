#
# ev3.py: An elitist (mu+mu) generational-with-overlap EA
# To run: python ev3.py --input config.cfg
#

import optparse
import sys
import yaml
import math
from random import Random
from Population import *
from utilities.utils import *
from utilities.ev_config import EV_Config

def evolution(cfg, obs_seq):
	#start random number generators
	uniprng = Random()
	uniprng.seed(cfg.randomSeed)
	normprng = Random()
	normprng.seed(cfg.randomSeed+101)

	#set static params on classes
	# (probably not the most elegant approach, but let's keep things simple...)
	Individual.minLimit = cfg.minLimit
	Individual.maxLimit = cfg.maxLimit
	Individual.N = 2
	Individual.M = 3
	Individual.uniprng = uniprng
	Individual.normprng = normprng
	Population.uniprng = uniprng
	Population.crossoverFraction = cfg.crossoverFraction
	  
	
	#create initial Population (random initialization)
	population=Population(cfg.populationSize)
	population.evaluateFitness(obs_seq)		
		
	#print initial pop stats	
	printStats(population,0)

	#accumulate & print stats
	stats = EV_Stats()
	stats.accumulate(population)
	stats.print()

	#evolution main loop
	for i in range(cfg.generationCount):
		#create initial offspring population by copying parent pop
		offspring=population.copy()
		
		#select mating pool
		offspring.conductTournament()

		#perform crossover
		offspring.crossover()
		
		#random mutation
		offspring.mutate()
		
		#update fitness values
		offspring.evaluateFitness(obs_seq)		
			
		#survivor selection: elitist truncation using parents+offspring
		population.combinePops(offspring)
		population.truncateSelect(cfg.populationSize)
		
		#print population stats	
		printStats(population,i+1)
		
		#accumulate & print stats    
		stats.accumulate(population)
		stats.print()
	#plot accumulated stats to file/screen using matplotlib
	stats.plot()
		
#
# Main entry point
#
def main(argv=None):
	if argv is None:
		argv = sys.argv
		
	try:
		#
		# get command-line options
		#
		parser = optparse.OptionParser()
		parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
		parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
		parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
		(options, args) = parser.parse_args(argv)
		
		#validate options
# 		if options.inputFileName is None:
# 			raise Exception("Must specify input file name using -i or --input option.")
		
		#Get EV3 config params
		cfg = EV_Config(options.inputFileName)
		
		#print config params
		print(cfg)

		#obs_seq = ['3L', '2M', '1S', '2M', '1S', '3L', '3L', '3L']
		obs_seq = ['o1', 'o1', 'o1', 'o2', 'o1', 'o3', 'o3', 'o3']
					
		#run EV3
		evolution(cfg, obs_seq)
		
		if not options.quietMode:					
			print('EV3 Completed!')	
	
	except Exception as info:
		if 'options' in vars() and options.debugMode:
			from traceback import print_exc
			print_exc()
		else:
			print(info)
	

if __name__ == '__main__':
	main(['-i', 'config.cfg', '-d', 'True'])
