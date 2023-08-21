#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
from Bio import SeqIO
from itertools import product
from scipy.fftpack import fft, ifft
import warnings
import sys
import scipy.stats
import statistics
import os
import io
from datetime import timedelta, datetime
import collections
import multiprocessing
warnings.filterwarnings("ignore")


#############################################################################
#############################################################################


def sequence_length(finput):
    length = 0
    for seq_record in SeqIO.parse(io.StringIO(finput), "fasta"):
        seq = seq_record.seq
        if len(seq) > length:
            length = len(seq)
    return length

        
def file_record(foutput, name_seq, mapping, label_dataset):
    dataset = open(foutput, 'a')
    dataset.write("nameseq,")
    for head in mapping:
        dataset.write("%s," % head)

    dataset.write("label")
    dataset.write("\n")
    dataset.write("%s," % (str(name_seq)))
    for map in mapping:
        dataset.write("%s," % map)
        # dataset.write("{0:.4f},".format(metric))
    dataset.write(label_dataset)
    dataset.write("\n")
    print("Recorded Sequence: %s" % name_seq)
    return


def chunksTwo(seq, win):
    seqlen = len(seq)
    for i in range(seqlen):
        j = seqlen if i+win>seqlen else i+win
        yield seq[i:j]
        if j==seqlen: break
    return


def classifical_chaos(finput, label_dataset, foutput):
    max_length = sequence_length(finput)
    for seq_record in SeqIO.parse(finput, "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        Sx = []
        Sy = []
        for nucle in seq:
            if nucle == "A":
                Sx.append(1)
                Sy.append(1)
            elif nucle == "C":
                Sx.append(-1)
                Sy.append(-1)
            elif nucle == "T" or nucle == "U":
                Sx.append(-1)
                Sy.append(1)
            else:
                Sx.append(1)
                Sy.append(-1)
        CGR_x = []
        CGR_y = []
        for i in range(0,len(Sx)):
            if i == 0:
                CGR_x.append(0.5 * Sx[i])
                CGR_y.append(0.5 * Sy[i])
            else:
                CGR_x.append(0.5 * Sx[i] + 0.5 * CGR_x[i - 1])
                CGR_y.append(0.5 * Sy[i] + 0.5 * CGR_y[i - 1])
        mapping = CGR_x + CGR_y
        padding = (max_length - len(Sx)) * 2
        mapping = np.pad(mapping, (0, padding), 'constant')
        file_record(foutput, name_seq, mapping, label_dataset)
    return foutput

#############################################################################
#############################################################################   
if __name__ == "__main__":
    print("\n")
    print("###################################################################################")
    print("##########            Feature Extraction: Chaos Game Theory             ###########")
    print("##########  Arguments: -i number of datasets -o output -r representation  #########")
    print("##########      -r:  1 = Classifical Chaos Game Representation            #########")
    print("##########               Author: Robson Parmezan Bonidia                ###########")
    print("###################################################################################")
    print("\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', help='Fasta format file | Number of dataset or labels')
    parser.add_argument('-o', '--output', help='Csv format file | E.g., train.csv')
    parser.add_argument('-r', '--approach', help='1 = Classifical Chaos Game Representation')
    args = parser.parse_args()
    n = int(args.number)
    foutput = str(args.output)
    representation = int(args.approach)
    dataset_labels = {}
    for i in range(1, n + 1):
        name = input("Dataset %s: " % i)
        label = input("Label for %s: " % name)
        print("\n")
        dataset_labels[name] = label
    max_length = sequence_length()
    if representation == 1:
        classifical_chaos()
    else:
        print("This package does not contain this approach - Check parameter -r")
#############################################################################
#############################################################################
