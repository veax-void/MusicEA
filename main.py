#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:38 2020

main.py: An elitist (mu+mu) generational-with-overlap EA ???
To run: python main.py -i config.cfg

"""
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import argparse
from utilities.ev_config import EV_Config
from EV import EV

def init_flags_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputFile',
					 action='store', help='File with configuration')

# 	parser.add_argument('-d', '--inputData', action='store', help='File with data')

	parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode',
					 default=False)
	return parser


def main(args = None):
	print(args)
	parser = init_flags_parser()
	if isinstance(args, list):
		# Parse args passed to the func
		args = parser.parse_args(args)
	else:
		# Parse args from terminal
		args = parser.parse_args()

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
	observed_sequence = ['o1', 'o1', 'o1', 'o2', 'o1', 'o3', 'o3', 'o3']

 	#run evolution
	ev = EV(cfg, observed_sequence)
	ev.run()


if __name__ == '__main__':
 	main(['-i', 'config.cfg'])
