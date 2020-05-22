#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
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
	parser.add_argument('-c','--inputFile', type=str, default=None, help='configuration file')
	parser.add_argument('-m', '--inputData', type=str, default=None, help='midi file')
	parser.add_argument('-q', '--quiet', action = "store_true", default = False, help='quiet mode')
	return parser


def main(args=None):
	parser = init_flags_parser()

	if isinstance(args, list):
		# Parse args passed to the main function
		args = parser.parse_args(args)
	else:
		# Parse args from terminal
		args = parser.parse_args()

	if not args.inputFile:
		raise Exception("Input config file not spesified! Use -c <filename>")
	if not args.inputData:
		raise Exception("Input config file not spesified! Use -c <filename>")


	#Get EV3 config params
	cfg = EV_Config(os.path.join(path, args.inputFile))

	#print config params
	print(cfg)

	csv_file = midi.midi_to_csv(args.inputData)
	notes = midi.csv_to_notes(csv_file)
	obs_state = list(np.unique(notes))
	observed_sequence = list(notes)

 	#run evolution
	ev = EV(cfg, observed_sequence, obs_state)
	stats = ev.run()

	with open("stat_gen{}_hid{}_state{}.pickle".format(cfg.generationCount, cfg.nHiddenStates, len(obs_state)), 'wb') as f:
		pickle.dump(stats, f)

if __name__ == '__main__':
	start = time.time()
	main()
	print("[INFO] Finished in {} hours".format((time.time() - start)/3600))
