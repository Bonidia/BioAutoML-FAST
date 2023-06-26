#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import numpy as np
import pandas as pd
import sys 
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/repDNA/')
from nac import *
from psenac import *
from ac import *
from Bio import SeqIO


def feature_extraction_repdna(finput, label):

	names_seq = []
	for seq_record in SeqIO.parse(finput, "fasta"):
		name = seq_record.name
		names_seq.append(name)

	rev_kmer = RevcKmer(k=3, normalize=True, upto=True)
	data_kmer = rev_kmer.make_revckmer_vec(open(finput))
	data_kmer = pd.DataFrame(data_kmer)
	
	psednc = PseDNC()
	data_psednc = psednc.make_psednc_vec(open(finput))
	data_psednc = pd.DataFrame(data_psednc)

	pseknc = PseKNC()
	data_pseknc = pseknc.make_pseknc_vec(open(finput))
	data_pseknc = pd.DataFrame(data_pseknc)

	sc_psednc = SCPseDNC()
	data_sc_psednc = sc_psednc.make_scpsednc_vec(open(finput), all_property=True)
	data_sc_psednc = pd.DataFrame(data_sc_psednc)

	sc_psetnc = SCPseTNC(lamada=2, w=0.05)
	data_sc_psetnc = sc_psetnc.make_scpsetnc_vec(open(finput), all_property=True)
	data_sc_psetnc = pd.DataFrame(data_sc_psetnc)

	dac = DAC(2)
	data_dac = dac.make_dac_vec(open(finput), all_property=True)
	data_dac = pd.DataFrame(data_dac)

	tac = TAC(2)
	data_tac = tac.make_tac_vec(open(finput), all_property=True)
	data_tac = pd.DataFrame(data_tac)

	tcc = TCC(2)
	data_tcc = tcc.make_tcc_vec(open(finput), all_property=True)
	data_tcc = pd.DataFrame(data_tcc)

	tacc = TACC(2)
	data_tacc = tacc.make_tacc_vec(open(finput), all_property=True)
	data_tacc = pd.DataFrame(data_tacc)

	df = pd.concat([data_kmer,
					data_psednc,
					data_pseknc,
					data_sc_psednc,
					data_sc_psetnc,
					data_dac,
					data_tac,
					data_tcc,
					data_tacc],
				    axis=1, ignore_index=False)

	df = df.add_prefix('repDNA_')
	df.insert(0, "nameseq", names_seq)
	df.insert(len(df.columns), "label", label)
	
	return df


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

	df = feature_extraction_repdna(input_file, label)
	df.to_csv(output_file, index=False, mode='a')

# Documentation: http://bioinformatics.hitsz.edu.cn/repDNA/static/download/repDNA_manual.pdf
	