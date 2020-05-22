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
from random import Random
from utilities.ev_config import EV_Config
from EV import EV
import utilities.midi_utils as midi
from Individual import Individual
from utilities.HMM import HiddenMarkovModel as HMM

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--inputMidi', type=str, default=None, help='input midi file')
	parser.add_argument('-b', '--bestModel', type=str, default=None, help='best model parameters')
	parser.add_argument('-s', '--save', type=str, default=None, help='generated midi file name')
	parser.add_argument('-q', '--quiet', action = "store_true", default = False, help='quiet mode')
	args = parser.parse_args()
	return args
	
args = init_flags_parser()

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
