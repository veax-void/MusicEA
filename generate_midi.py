#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020
"""
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import numpy as np
import pickle
import argparse
from random import Random

import utilities.midi_utils as midi
from Individual import Individual

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--inputMidi', type=str, default=None, help='input midi file')
	parser.add_argument('-b', '--bestModel', type=str, default=None, help='best model parameters')
	parser.add_argument('-s', '--saveName', type=str, default=None, help='generated midi file name')
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

	if not args.inputMidi:
		raise Exception("Input midi file not spesified! Use -m <filename>")
	if not args.bestModel:
		raise Exception("Input model file not spesified! Use -b <filename>")
	if not args.saveName:
		raise Exception("Input name of outfile not spesified! Use -s <filename>")


	csv_file = midi.midi_to_csv(args.inputMidi)
	notes = midi.csv_to_notes(csv_file)

	with open(args.bestModel, 'rb') as f:
		stats = pickle.load(f)
	stats.plot()
	bestState = stats.bestState[-1]

	Individual.nHiddenStates = bestState[0].shape[0]
	Individual.nObservableStates = bestState[2].shape[1]
	uniprng = Random()
	Individual.uniprng = uniprng
	Individual.normprng = uniprng
	Individual.observed_sequence = notes
	bestInd = Individual()

	genNotes, _ = bestInd.model.generate(len(notes))
	genNotes = np.array(genNotes)
	midi.notes_to_midi(args.inputMidi, genNotes, args.save)

if __name__ == '__main__':
	main()