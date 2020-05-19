#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:41:59 2020

"""
from midi_utils import midi_to_csv, csv_to_notes, notes_to_midi
import numpy as np
import pickle
from HMM import HiddenMarkovModel as HMM

def extract_notes(filename):
	midi_filename = filename

	csv_midi = midi_to_csv(midi_filename)
	print(csv_midi.shape, csv_midi, '\n')

	notes = csv_to_notes(csv_midi)
	print(notes.shape, notes, '\n')
	
	new_mini_name = 'ten_notes.mid'
	ten_notes = np.zeros(notes.shape, dtype='int') + 10
	ten_notes = np.array(ten_notes, dtype='str')
	
	midi_file = notes_to_midi(midi_filename, ten_notes, new_mini_name)
	
	return csv_midi, notes
#if __name__ == '__main__':
csv_midi, notes = extract_notes('../data/bwv988.mid')
with open("best_hmm.pickle", 'rb') as f:
	state = pickle.load(f)

[pi, A, B] = state
st = ['h{}'.format(i+1) for i in range(len(pi))]
ob = ['{}'.format(i) for i in range(128)]
model = HMM(st, ob)
model.set_pi(pi)
model.set_A(A)
model.set_B(B)

sample = model.generate(len(notes))

