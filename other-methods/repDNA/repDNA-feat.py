#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import itertools
import numpy as np
import pandas as pd
import re
import sys 
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/repDNA/')
from nac import *
from psenac import *
from ac import *
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import cpu_count


DINUCLEOTIDE_PROPERTIES = [
	'Base stacking', 'Protein induced deformability', 'B-DNA twist',
	'Dinucleotide GC Content', 'A-philicity', 'Propeller twist',
	'Duplex stability:(freeenergy)', 'Duplex tability(disruptenergy)',
	'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
	'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG',
	'Breslauer_dH', 'Breslauer_dS', 'Electron_interaction',
	'Hartman_trans_free_energy', 'Helix-Coil_transition',
	'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction',
	'SantaLucia_dG', 'SantaLucia_dH', 'SantaLucia_dS',
	'Sarai_flexibility', 'Stability', 'Stacking_energy', 'Sugimoto_dG',
	'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist',
	'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'
]

TRINUCLEOTIDE_PROPERTIES = [
	'Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)',
	'Trinucleotide GC Content', 'Nucleosome positioning', 'Consensus_roll',
	'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons', 'MW-kg',
	'Nucleosome', 'Nucleosome-Rigid'
]


def feature_slug(value):
	"""Return a stable, readable token for a feature name."""

	return re.sub(r'_+', '_', re.sub(r'[^A-Za-z0-9]+', '_', value)).strip('_')


def kmers(k):
	return [''.join(parts) for parts in itertools.product('ACGT', repeat=k)]


def reverse_complement(sequence):
	return sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def descriptor_feature_names(name):
	"""Return semantic names in the exact order emitted by repDNA."""

	if name == 'Revkmer':
		features = []
		for k in range(1, 4):
			for kmer in kmers(k):
				canonical = min(kmer, reverse_complement(kmer))
				if canonical == kmer:
					features.append(f'repDNA__Revkmer__k{k}__{canonical}')
		return features
	if name == 'PseDNC':
		return ([f'repDNA__PseDNC__composition__{kmer}' for kmer in kmers(2)] +
				[f'repDNA__PseDNC__theta__lag_{lag}' for lag in range(1, 4)])
	if name == 'PseKNC':
		return ([f'repDNA__PseKNC__composition__{kmer}' for kmer in kmers(3)] +
				['repDNA__PseKNC__theta__lag_1'])
	if name == 'SC-PseDNC':
		return ([f'repDNA__SC-PseDNC__composition__{kmer}' for kmer in kmers(2)] + [
			f'repDNA__SC-PseDNC__theta__lag_1__{feature_slug(prop)}'
			for prop in DINUCLEOTIDE_PROPERTIES
		])
	if name == 'SC-PseTNC':
		return ([f'repDNA__SC-PseTNC__composition__{kmer}' for kmer in kmers(3)] + [
			f'repDNA__SC-PseTNC__theta__lag_{lag}__{feature_slug(prop)}'
			for lag in range(1, 3) for prop in TRINUCLEOTIDE_PROPERTIES
		])
	if name == 'DAC':
		return [
			f'repDNA__DAC__lag_{lag}__{feature_slug(prop)}'
			for lag in range(1, 3) for prop in DINUCLEOTIDE_PROPERTIES
		]
	if name == 'TAC':
		return [
			f'repDNA__TAC__lag_{lag}__{feature_slug(prop)}'
			for lag in range(1, 3) for prop in TRINUCLEOTIDE_PROPERTIES
		]
	if name == 'TCC':
		return [
			f'repDNA__TCC__lag_{lag}__{feature_slug(left)}__to__{feature_slug(right)}'
			for lag in range(1, 3)
			for left in TRINUCLEOTIDE_PROPERTIES
			for right in TRINUCLEOTIDE_PROPERTIES
			if left != right
		]
	if name == 'TACC':
		return [
			f'repDNA__TACC__auto__lag_{lag}__{feature_slug(prop)}'
			for lag in range(1, 3) for prop in TRINUCLEOTIDE_PROPERTIES
		] + [
			f'repDNA__TACC__cross__lag_{lag}__{feature_slug(left)}__to__{feature_slug(right)}'
			for lag in range(1, 3)
			for left in TRINUCLEOTIDE_PROPERTIES
			for right in TRINUCLEOTIDE_PROPERTIES
			if left != right
		]
	raise ValueError(f'Unknown repDNA descriptor: {name}')

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

# Combination of TAC and TCC used by the published feature schema.
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
	parser.add_argument("--n_cpu", type=int, default=-1, help="Maximum descriptor workers; default respects container CPU limits")
	parser.add_argument("--descriptors", nargs='+',
						choices=['Revkmer', 'PseDNC', 'PseKNC', 'SC-PseDNC',
								 'SC-PseTNC', 'DAC', 'TAC', 'TCC', 'TACC'])
	args = parser.parse_args()
	input_file = str(args.file)
	label = str(args.labelFile)
	output_file = str(args.outFile)

	names_seq = []
	for seq_record in SeqIO.parse(input_file, "fasta"):
		name = seq_record.name
		names_seq.append(name)

	descriptors = [
		('Revkmer', revkmer),
		('PseDNC', psednc),
		('PseKNC', pseknc),
		('SC-PseDNC', sc_psednc),
		('SC-PseTNC', sc_psetnc),
		('DAC', dac),
		('TAC', tac),
		('TCC', tcc),
		('TACC', tacc),
	]
	if args.descriptors:
		descriptors = [descriptor for descriptor in descriptors if descriptor[0] in args.descriptors]

	# Preallocate result list
	results = [None] * len(descriptors)

	# Run descriptors in parallel
	available_cpus = cpu_count()
	n_cpu = available_cpus if args.n_cpu < 1 else min(args.n_cpu, available_cpus)
	with ProcessPoolExecutor(max_workers=min(len(descriptors), n_cpu)) as executor:
		futures = [
			executor.submit(run_descriptor, (i, func, input_file))
			for i, (_, func) in enumerate(descriptors)
		]

		for future in as_completed(futures):
			idx, name, res = future.result()
			print(name, len(res.columns))
			descriptor_name = descriptors[idx][0]
			feature_names = descriptor_feature_names(descriptor_name)
			if len(feature_names) != len(res.columns):
				raise ValueError(
					f'{descriptor_name} emitted {len(res.columns)} columns; '
					f'expected {len(feature_names)}.'
				)
			res.columns = feature_names
			results[idx] = res  # order preserved

	# Concatenate in correct order
	df = pd.concat(results, axis=1, ignore_index=False)

	# Insert metadata
	df.insert(0, "nameseq", names_seq)
	df["label"] = label

	df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
# Documentation: http://bioinformatics.hitsz.edu.cn/repDNA/static/download/repDNA_manual.pdf
