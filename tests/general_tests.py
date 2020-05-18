#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:41:59 2020

"""
from midi_utils import midi_to_csv, csv_to_notes, notes_to_midi
from HMM import HiddenMarkovModel
import numpy as np

def main():
    midi_filename = 'bwv988.mid'
    
    csv_midi = midi_to_csv(midi_filename)
    print(csv_midi.shape, csv_midi, '\n')

    notes = csv_to_notes(csv_midi)
    print(notes.shape, notes, '\n')
    
    new_mini_name = 'ten_notes.mid'
    ten_notes = np.zeros(notes.shape, dtype='int') + 10
    ten_notes = np.array(ten_notes, dtype='str')
    
    midi_file = notes_to_midi(midi_filename, ten_notes, new_mini_name)
    
    
    
    # Init stuff
    states = ['1H', '2C']
    observables = ['1S', '2M', '3L']
    
    observations = []
    for _ in range(7600):
    	observations.append(np.random.choice(observables))

    # Define Hidden Markov Model
    hmm = HiddenMarkovModel(states, observables)

    # Get score for observations
    score = hmm.score(observations)
    print("Score: {}\n".format(score))

    
    
if __name__ == '__main__':
    main()