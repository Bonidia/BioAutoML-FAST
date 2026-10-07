#!/usr/bin/env python
#_*_coding:utf-8_*_

import argparse
import os
import re
import numpy as np
import pandas as pd
from codes import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.",
									 description="Generating various numerical representation schemes for protein sequences")
	parser.add_argument("--file", required=True, help="input fasta file")
	parser.add_argument("--type", required=True, nargs='+',
						choices=['All', 'CKSAAP', 'DDE',
								 'GAAC', 'CKSAAGP', 'GDPC', 'GTPC',
								 'CTDC', 'CTDT', 'CTDD',
								 'CTriad', 'KSCTriad'],
						help="the encoding type")
	parser.add_argument("--path", dest='filePath',
						help="data file path used for 'PSSM', 'SSEB(C)', 'Disorder(BC)', 'ASA' and 'TA' encodings")
	parser.add_argument("--train", dest='trainFile',
						help="training file in fasta format only used for 'KNNprotein' or 'KNNpeptide' encodings")
	parser.add_argument("--label", dest='labelFile',
						help="sample label file only used for 'KNNprotein' or 'KNNpeptide' encodings")
	parser.add_argument("--order", dest='order',
						choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'],
						help="output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descriptors")
	parser.add_argument("--userDefinedOrder", dest='userDefinedOrder',
						help="user defined output order for of Amino Acid Composition (i.e. AAC, EAAC, CKSAAP, DPC, DDE, TPC) descriptors")
	parser.add_argument("--out", dest='outFile',
						help="the generated descriptor file")
	args = parser.parse_args()
	fastas = readFasta.readFasta(args.file)
	userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder != None else 'ACDEFGHIKLMNPQRSTVWY'
	userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
	if len(userDefinedOrder) != 20:
		userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
	myAAorder = {
		'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
		'polarity': 'DENKRQHSGTAPYVMCWIFL',
		'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
		'userDefined': userDefinedOrder
	}
	myOrder = myAAorder[args.order] if args.order != None else 'ACDEFGHIKLMNPQRSTVWY'
	kw = {'path': args.filePath, 'train': args.trainFile, 'label': args.labelFile, 'order': myOrder}
	label = str(args.labelFile)
	output = str(args.outFile)

	descriptor_names = [
		'CKSAAP', 'DDE', 'GAAC', 'CKSAAGP', 'GDPC', 'GTPC',
		'CTDC', 'CTDT', 'CTDD', 'CTriad', 'KSCTriad'
	]
	selected_names = descriptor_names if 'All' in args.type else [
		name for name in descriptor_names if name in args.type
	]

	descriptor_frames = []
	for name in selected_names:
		descriptor = eval(name + '.' + name + '(fastas, **kw)')
		descriptor = pd.DataFrame(descriptor[1:], columns=descriptor[0])
		descriptor.rename(
			columns={column: f'{name}__{column}' for column in descriptor.columns if column != '#'},
			inplace=True
		)
		if descriptor_frames:
			descriptor = descriptor.iloc[:, 1:]
		descriptor_frames.append(descriptor)

	df = pd.concat(descriptor_frames, axis=1, ignore_index=False)
	df.rename(columns={"#": "nameseq"}, inplace=True)
	df.insert(len(df.columns), "label", label)


	df.to_csv(output, index=False, mode='a', header=not os.path.exists(output))
	# print(output)
	# print(df)
	# print(label)
	# desc_cksaap.to_csv('test.csv', index=False)
	# outFile = args.outFile if args.outFile != None else 'encoding.csv'
	# saveCode.savetsv(encodings, outFile)
	# python iFeature/iFeature.py --file 1-pvp/TSpos63.fasta --type All --label test --out test2.csv
