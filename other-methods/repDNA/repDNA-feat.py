#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import numpy as np
import pandas as pd
import sys 
import os
import multiprocessing as mp
from concurrent import futures
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/repDNA/')
from nac import *
from psenac import *
from ac import *
from Bio import SeqIO


def revkmer(finput):
	rev_kmer = RevcKmer(k=3, normalize=True, upto=True)
	data_kmer = rev_kmer.make_revckmer_vec(open(finput))
	return pd.DataFrame(data_kmer)
	

def psednc(finput):
	psednc = PseDNC()
	data_psednc = psednc.make_psednc_vec(open(finput))
	return pd.DataFrame(data_psednc)


def pseknc(finput):
	pseknc = PseKNC()
	data_pseknc = pseknc.make_pseknc_vec(open(finput))
	return pd.DataFrame(data_pseknc)
 

def sc_psednc(finput):
    sc_psednc = SCPseDNC()
    data_sc_psednc = sc_psednc.make_scpsednc_vec(open(finput), all_property=True)
    return pd.DataFrame(data_sc_psednc)


def sc_psetnc(finput):
    sc_psetnc = SCPseTNC(lamada=2, w=0.05)
    data_sc_psetnc = sc_psetnc.make_scpsetnc_vec(open(finput), all_property=True)
    return pd.DataFrame(data_sc_psetnc)

def dac(finput):
	dac = DAC(2)
	data_dac = dac.make_dac_vec(open(finput), all_property=True)
	return pd.DataFrame(data_dac)


def tac(finput):
	tac = TAC(2)
	data_tac = tac.make_tac_vec(open(finput), all_property=True)
	return pd.DataFrame(data_tac)


def tcc(finput):
	tcc = TCC(2)
	data_tcc = tcc.make_tcc_vec(open(finput), all_property=True)
	return pd.DataFrame(data_tcc)


def tacc(finput):
	tacc = TACC(2)
	data_tacc = tacc.make_tacc_vec(open(finput), all_property=True)
	return pd.DataFrame(data_tacc)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", dest='file')
	parser.add_argument("--output", dest='outFile',
						help="the generated descriptor file")
	parser.add_argument("--label", dest='labelFile')
	args = parser.parse_args()
	input_file = str(args.file)
	label = str(args.labelFile)
	output_file = str(args.outFile)
	
	names_seq = []
	for seq_record in SeqIO.parse(input_file, "fasta"):
		name = seq_record.name
		names_seq.append(name)

	results = []
	descriptors = [revkmer, psednc, pseknc, sc_psednc, sc_psetnc]
 
	# executor = futures.ProcessPoolExecutor()
	
	with futures.ThreadPoolExecutor(max_workers=mp.cpu_count() - 2) as executor:
		runs = [executor.submit(i, input_file) for i in descriptors]
		futures.wait(runs)
		for data in range(0, len(runs)):
			results.append(runs[data].result())
		print(results[1])
 
	# executor.submit(revkmer, input_file)
	# executor.submit(psednc, input_file)
	# executor.shutdown(wait=True)
	# print(results)
	
	df = pd.concat([res for res in results], axis=1, ignore_index=False)
	new_names = []
	for col in range(0, len(df.columns)):
		new_names.append('repDNA-' + str(col))
	# df = df.add_prefix('repDNA_')
	df = df.set_axis(new_names, axis=1)
	df.insert(0, "nameseq", names_seq)
	df.insert(len(df.columns), "label", label)
	df.to_csv(output_file, index=False, mode='a')

# Documentation: http://bioinformatics.hitsz.edu.cn/repDNA/static/download/repDNA_manual.pdf
	