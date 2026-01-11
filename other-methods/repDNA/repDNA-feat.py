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
from concurrent.futures import ProcessPoolExecutor, as_completed

# 	A variant of the basic kmer, in which the kmers are not expected to be strand-specific, so reverse complementary are collapsed into a single feature
def revkmer(finput):
	rev_kmer = RevcKmer(k=3, normalize=True, upto=True)
	data_kmer = rev_kmer.make_revckmer_vec(open(finput))
	return pd.DataFrame(data_kmer)
	
# Combining dinucleotide composition and global sequence-order effects
def psednc(finput):
	psednc = PseDNC()
	data_psednc = psednc.make_psednc_vec(open(finput))
	return pd.DataFrame(data_psednc)

# Improving PseDNC by incorporating k-tuple nucleotide composition
def pseknc(finput):
	pseknc = PseKNC()
	data_pseknc = pseknc.make_pseknc_vec(open(finput))
	return pd.DataFrame(data_pseknc)
 
# Combining dinucleotide composition and global sequence-order effects by series correlation
def sc_psednc(finput):
    sc_psednc = SCPseDNC()
    data_sc_psednc = sc_psednc.make_scpsednc_vec(open(finput), all_property=True)
    return pd.DataFrame(data_sc_psednc)

# Combining trinucleotide composition and global sequence-order effects by series correlation
def sc_psetnc(finput):
    sc_psetnc = SCPseTNC(lamada=2, w=0.05)
    data_sc_psetnc = sc_psetnc.make_scpsetnc_vec(open(finput), all_property=True)
    return pd.DataFrame(data_sc_psetnc)

# Incorporating the correlation of the same property between two dinucleotides
def dac(finput):
	dac = DAC(2)
	data_dac = dac.make_dac_vec(open(finput), all_property=True)
	return pd.DataFrame(data_dac)

# Incorporating the correlation of the same property between two trinucleotides
def tac(finput):
	tac = TAC(2)
	data_tac = tac.make_tac_vec(open(finput), all_property=True)
	return pd.DataFrame(data_tac)

# Incorporating the correlation of the different properties between two trinucleotides
def tcc(finput):
	tcc = TCC(2)
	data_tcc = tcc.make_tcc_vec(open(finput), all_property=True)
	return pd.DataFrame(data_tcc)

# Combination of TAC and TCC
def tacc(finput):
	tacc = TACC(2)
	data_tacc = tacc.make_tacc_vec(open(finput), all_property=True)
	return pd.DataFrame(data_tacc)

def run_descriptor(idx_func):
    idx, func, input_file = idx_func
    res = func(input_file)
    return idx, func.__name__, res

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

	descriptors = [
		revkmer, psednc, pseknc,
		sc_psednc, sc_psetnc,
		dac, tac, tcc, tacc
	]

	# Preallocate result list
	results = [None] * len(descriptors)

	# Run descriptors in parallel
	with ProcessPoolExecutor() as executor:
		futures = [
			executor.submit(run_descriptor, (i, func, input_file))
			for i, func in enumerate(descriptors)
		]

		for future in as_completed(futures):
			idx, name, res = future.result()
			print(name, len(res.columns))
			results[idx] = res  # order preserved

	# Concatenate in correct order
	df = pd.concat(results, axis=1, ignore_index=False)

	# Rename columns
	df.columns = [f"repDNA-{i}" for i in range(len(df.columns))]

	# Insert metadata
	df.insert(0, "nameseq", names_seq)
	df["label"] = label

	df.to_csv(output_file, index=False, mode='a')
# Documentation: http://bioinformatics.hitsz.edu.cn/repDNA/static/download/repDNA_manual.pdf
	