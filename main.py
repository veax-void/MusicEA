#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020

main.py: An elitist (mu+mu) generational-with-overlap EA ???
To run: python main.py -i config.cfg

"""
import time
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import numpy as np
import pickle
import argparse
from utilities.ev_config import EV_Config
from EV import EV
import utilities.midi_utils as midi

def init_flags_parser():
	parser = argparse.ArgumentParser()
	#parser.add_argument('-c', '--inputFile', action='store', help='File with configuration')
 	#parser.add_argument('-d', '--inputData', action='store', help='File with data')
	parser.add_argument('-c','--inputFile', type=str, default=None, help='configuration file')
	parser.add_argument('-m', '--inputData', type=str, default=None, help='midi file')
	parser.add_argument('-q', '--quiet', action = "store_true", default = False, help='quiet mode')
	args = parser.parse_args()
	return args


def main(args = None):
	args = init_flags_parser()

	if args.inputFile:
		config_filename = args.inputFile
	else:
		raise Exception("Input config file not spesified! Use -i <filename>")


	#Get EV3 config params
	cfg = EV_Config(os.path.join(path,config_filename))

	#print config params
	print(cfg)


	# Bad decision
# 		#obs_seq = ['3L', '2M', '1S', '2M', '1S', '3L', '3L', '3L']
	#observed_sequence = ['o1', 'o1', 'o1', 'o2', 'o1', 'o3', 'o3', 'o3']
	csv_file = midi.midi_to_csv(args.inputData)
	notes = midi.csv_to_notes(csv_file)
	obs_state = np.unique(notes)
	observed_sequence = list(notes) 

 	#run evolution
	ev = EV(cfg, observed_sequence, obs_state)
	stats = ev.run()

	with open("stat_gen{}_hid{}.pickle".format(cfg.generationCount, cfg.nHiddenStates), 'wb') as f:
		pickle.dump(stats, f)

if __name__ == '__main__':
	start = time.time()
	main()
	print("[INFO] Finished in {} hours".format((time.time() - start)/3600))
