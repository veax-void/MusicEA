#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:50:37 2020
"""
import sys,os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import py_midicsv
import numpy as np
import pandas as pd

def midi_to_csv(midi_filename:str) -> pd.DataFrame:
    csv_string = py_midicsv.midi_to_csv(midi_filename)

    data = []
    for row in csv_string:
        row = row.replace('\n','')
        row = row.split(', ')
        data += [row]

    return pd.DataFrame(data)


def csv_to_notes(data:pd.DataFrame) -> np.ndarray:
    cond = data[2] == 'Note_on_c'
    notes = data[cond][4]
    return notes.to_numpy()

def notes_to_midi(midi_filename:str, notes:np.ndarray, new_mini_name:str):
    csv_string = py_midicsv.midi_to_csv(midi_filename)

    data = []
    k = 0
    for i in range(len(csv_string)):
        row = csv_string[i]
        row = row.replace('\n','')
        row = row.split(', ')

        if row[2] == 'Note_on_c':
            row[4] = str(notes[k])
            k += 1
            csv_string[i] = ', '.join(row) + '\n'

    midi_object = py_midicsv.csv_to_midi(csv_string)

    with open(new_mini_name+'.mid', "wb") as output_file:
        midi_writer = py_midicsv.FileWriter(output_file)
        midi_writer.write(midi_object)
